/*
  vibes - a transformer language model. pure c++17. zero deps.
  backprop: analytical (real gradients, not finite diff)
  tokenizer: byte-pair encoding
  sampling: temperature + top-k + top-p + repetition penalty + beam search
  training: adamw + cosine lr + warmup + grad clip + dropout + batch
  extras: lora adapters, checkpointing, personality injection
  personality: baked in at generation time, not trained in
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include <cassert>
#include <memory>
#include <functional>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <queue>

using namespace std;
using clk = chrono::high_resolution_clock;

// ─────────────────────────────────────────────
// rng
// ─────────────────────────────────────────────
static mt19937 RNG(chrono::steady_clock::now().time_since_epoch().count());
static bool TRAINING = false;
static float randn_f(){static normal_distribution<float> d(0,1);return d(RNG);}
static float randf(float a,float b){return uniform_real_distribution<float>(a,b)(RNG);}
static bool  flip(float p){return randf(0,1)<p;}

// ─────────────────────────────────────────────
// safe math
// ─────────────────────────────────────────────
static inline float safe_exp(float x){ return expf(max(-80.f,min(80.f,x))); }
static inline float safe_log(float x){ return logf(max(x,1e-10f)); }

// ─────────────────────────────────────────────
// Param: weight + gradient storage
// ─────────────────────────────────────────────
struct Param {
    vector<float> w, g;
    int rows, cols;
    string name;

    Param()=default;
    Param(int r,int c,float scale=0.02f,string n="")
        :w(r*c,0.f),g(r*c,0.f),rows(r),cols(c),name(move(n)){
        if(scale>0) for(auto& v:w) v=randn_f()*scale;
    }

    float  W(int i,int j) const { return w[i*cols+j]; }
    float& W(int i,int j)       { return w[i*cols+j]; }
    float& G(int i,int j)       { return g[i*cols+j]; }
    int numel() const { return rows*cols; }
    void zero_grad() { fill(g.begin(),g.end(),0.f); }

    void clip_weights(float maxv){
        for(auto& v:w) v=max(-maxv,min(maxv,v));
    }

    void save(ofstream& f) const {
        int n=w.size(); f.write((char*)&n,4);
        f.write((char*)w.data(),n*4);
    }
    void load(ifstream& f){
        int n; f.read((char*)&n,4);
        w.resize(n); g.assign(n,0.f);
        f.read((char*)w.data(),n*4);
    }
};

// ─────────────────────────────────────────────
// LoRA adapter (low-rank decomposition)
//   W_eff = W + (B @ A) * scale
//   only A,B are trained during lora fine-tune
// ─────────────────────────────────────────────
struct LoRA {
    Param A, B;    // A: (in, rank), B: (rank, out)
    float scale;
    int rank;
    bool active;

    LoRA():scale(1.f),rank(0),active(false){}
    LoRA(int in_f,int out_f,int r,float alpha=1.f)
        :A(in_f,r,1.f/sqrtf((float)r)),B(r,out_f,0.f),
         scale(alpha/(float)r),rank(r),active(true){}

    // delta = (x @ A) @ B * scale
    void fwd_delta(const float* x,int in_f,float* delta,int out_f) const {
        if(!active) return;
        vector<float> mid(rank,0.f);
        for(int r=0;r<rank;r++){
            float s=0;
            for(int i=0;i<in_f;i++) s+=x[i]*A.w[i*rank+r];
            mid[r]=s;
        }
        for(int j=0;j<out_f;j++){
            float s=0;
            for(int r=0;r<rank;r++) s+=mid[r]*B.w[r*out_f+j];
            delta[j]+=s*scale;
        }
    }

    // bwd through lora delta
    void bwd_delta(const float* x,int in_f,const float* dout,int out_f,float* dx){
        if(!active) return;
        vector<float> mid(rank,0.f);
        for(int r=0;r<rank;r++){
            float s=0;
            for(int i=0;i<in_f;i++) s+=x[i]*A.w[i*rank+r];
            mid[r]=s;
        }
        vector<float> dmid(rank,0.f);
        for(int r=0;r<rank;r++) for(int j=0;j<out_f;j++){
            B.g[r*out_f+j]+=mid[r]*dout[j]*scale;
            dmid[r]+=B.w[r*out_f+j]*dout[j]*scale;
        }
        for(int i=0;i<in_f;i++) for(int r=0;r<rank;r++){
            A.g[i*rank+r]+=x[i]*dmid[r];
            dx[i]+=A.w[i*rank+r]*dmid[r];
        }
    }

    void zero_grad(){ A.zero_grad(); B.zero_grad(); }
    void save(ofstream& f) const { int act=active; f.write((char*)&act,4); if(active){A.save(f);B.save(f);} }
    void load(ifstream& f){ int act; f.read((char*)&act,4); active=act; if(active){A.load(f);B.load(f);} }
};

// ─────────────────────────────────────────────
// AdamW optimizer
// ─────────────────────────────────────────────
struct AdamW {
    float lr,b1,b2,eps,wd,clip;
    int t;
    vector<Param*> ps;
    vector<vector<float>> m,v;

    AdamW(float lr_=3e-4f,float b1_=0.9f,float b2_=0.999f,
          float eps_=1e-8f,float wd_=0.1f,float clip_=1.f)
        :lr(lr_),b1(b1_),b2(b2_),eps(eps_),wd(wd_),clip(clip_),t(0){}

    void add(Param* p){ ps.push_back(p); m.push_back(vector<float>(p->numel(),0.f)); v.push_back(vector<float>(p->numel(),0.f)); }

    float grad_norm(){
        float s=0; for(auto* p:ps) for(float g:p->g) s+=g*g; return sqrtf(s);
    }

    void step(float lr_scale=1.f){
        t++;
        float gn=grad_norm(), c=(gn>clip&&clip>0)?clip/gn:1.f;
        float bc1=1.f-powf(b1,t), bc2=1.f-powf(b2,t), cur=lr*lr_scale;
        for(int pi=0;pi<(int)ps.size();pi++){
            auto* p=ps[pi];
            for(int i=0;i<p->numel();i++){
                float gi=p->g[i]*c;
                m[pi][i]=b1*m[pi][i]+(1-b1)*gi;
                v[pi][i]=b2*v[pi][i]+(1-b2)*gi*gi;
                float mh=m[pi][i]/bc1, vh=v[pi][i]/bc2;
                p->w[i]-=cur*(mh/(sqrtf(vh)+eps)+wd*p->w[i]);
            }
        }
    }

    void save(ofstream& f) const {
        f.write((char*)&t,4);
        for(int pi=0;pi<(int)m.size();pi++){
            int n=m[pi].size(); f.write((char*)&n,4);
            f.write((char*)m[pi].data(),n*4);
            f.write((char*)v[pi].data(),n*4);
        }
    }
    void load(ifstream& f){
        f.read((char*)&t,4);
        for(int pi=0;pi<(int)m.size();pi++){
            int n; f.read((char*)&n,4);
            m[pi].resize(n); v[pi].resize(n);
            f.read((char*)m[pi].data(),n*4);
            f.read((char*)v[pi].data(),n*4);
        }
    }
};

// ─────────────────────────────────────────────
// Linear layer with optional LoRA
// ─────────────────────────────────────────────
struct Linear {
    Param W, b;
    LoRA lora;
    bool has_bias;
    int in_f, out_f;
    mutable vector<float> x_cache;

    Linear()=default;
    Linear(int in,int out,bool bias=true,float scale=0.02f)
        :W(in,out,scale),b(1,out,0.f),has_bias(bias),in_f(in),out_f(out){}

    void attach_lora(int rank,float alpha=1.f){ lora=LoRA(in_f,out_f,rank,alpha); }

    void fwd1d(const vector<float>& x,vector<float>& o) const {
        x_cache=x; o.assign(out_f,0.f);
        for(int j=0;j<out_f;j++){
            float s=has_bias?b.w[j]:0.f;
            for(int i=0;i<in_f;i++) s+=x[i]*W.w[i*out_f+j];
            o[j]=s;
        }
        if(lora.active) lora.fwd_delta(x.data(),in_f,o.data(),out_f);
    }

    void bwd1d(const vector<float>& dout,vector<float>& dx){
        dx.assign(in_f,0.f);
        for(int i=0;i<in_f;i++) for(int j=0;j<out_f;j++){
            W.G(i,j)+=x_cache[i]*dout[j];
            dx[i]+=W.w[i*out_f+j]*dout[j];
        }
        if(has_bias) for(int j=0;j<out_f;j++) b.g[j]+=dout[j];
        if(lora.active) lora.bwd_delta(x_cache.data(),in_f,dout.data(),out_f,dx.data());
    }

    void fwd2d(const vector<float>& x,int T,vector<float>& o) const {
        x_cache=x; o.assign(T*out_f,0.f);
        for(int t=0;t<T;t++){
            const float* xt=x.data()+t*in_f;
            float* ot=o.data()+t*out_f;
            for(int j=0;j<out_f;j++){
                float s=has_bias?b.w[j]:0.f;
                for(int i=0;i<in_f;i++) s+=xt[i]*W.w[i*out_f+j];
                ot[j]=s;
            }
            if(lora.active) lora.fwd_delta(xt,in_f,ot,out_f);
        }
    }

    void bwd2d(const vector<float>& dout,int T,vector<float>& dx){
        dx.assign(T*in_f,0.f);
        for(int t=0;t<T;t++){
            const float* xt=x_cache.data()+t*in_f;
            const float* dt=dout.data()+t*out_f;
            float* dxt=dx.data()+t*in_f;
            for(int i=0;i<in_f;i++) for(int j=0;j<out_f;j++){
                W.G(i,j)+=xt[i]*dt[j];
                dxt[i]+=W.w[i*out_f+j]*dt[j];
            }
            if(has_bias) for(int j=0;j<out_f;j++) b.g[j]+=dt[j];
            if(lora.active) lora.bwd_delta(xt,in_f,dt,out_f,dxt);
        }
    }

    void reg(AdamW& opt){ opt.add(&W); if(has_bias) opt.add(&b); }
    void reg_lora(AdamW& opt){ if(lora.active){opt.add(&lora.A);opt.add(&lora.B);} }

    void save(ofstream& f) const { W.save(f); if(has_bias) b.save(f); lora.save(f); }
    void load(ifstream& f){ W.load(f); if(has_bias) b.load(f); lora.load(f); }
};

// ─────────────────────────────────────────────
// LayerNorm  (with backward)
// ─────────────────────────────────────────────
struct LayerNorm {
    Param gamma, beta;
    int dim; float eps;
    mutable vector<float> x_in,xhat,rstd;
    mutable int Tc=0;

    LayerNorm()=default;
    LayerNorm(int d,float ep=1e-5f):gamma(1,d,0.f),beta(1,d,0.f),dim(d),eps(ep){
        fill(gamma.w.begin(),gamma.w.end(),1.f);
    }

    void fwd(const vector<float>& x,int T,vector<float>& o) const {
        x_in=x; Tc=T; xhat.resize(T*dim); rstd.resize(T); o.resize(T*dim);
        for(int t=0;t<T;t++){
            const float* xt=x.data()+t*dim;
            float mean=0; for(int i=0;i<dim;i++) mean+=xt[i]; mean/=dim;
            float var=0; for(int i=0;i<dim;i++){float d=xt[i]-mean;var+=d*d;} var/=dim;
            float rs=1.f/sqrtf(var+eps); rstd[t]=rs;
            for(int i=0;i<dim;i++){
                float xh=(xt[i]-mean)*rs; xhat[t*dim+i]=xh;
                o[t*dim+i]=gamma.w[i]*xh+beta.w[i];
            }
        }
    }

    void bwd(const vector<float>& dout,vector<float>& dx){
        dx.assign(Tc*dim,0.f);
        for(int t=0;t<Tc;t++){
            const float* do_=dout.data()+t*dim;
            float* dxt=dx.data()+t*dim;
            float s1=0,s2=0;
            vector<float> dxh(dim);
            for(int i=0;i<dim;i++){
                dxh[i]=do_[i]*gamma.w[i];
                s1+=dxh[i]*xhat[t*dim+i]; s2+=dxh[i];
                gamma.g[i]+=do_[i]*xhat[t*dim+i];
                beta.g[i] +=do_[i];
            }
            float rs=rstd[t];
            for(int i=0;i<dim;i++)
                dxt[i]=rs*(dxh[i]-(s2+xhat[t*dim+i]*s1)/dim);
        }
    }

    void reg(AdamW& opt){ opt.add(&gamma); opt.add(&beta); }
    void save(ofstream& f) const { gamma.save(f); beta.save(f); }
    void load(ifstream& f){ gamma.load(f); beta.load(f); }
};

// ─────────────────────────────────────────────
// GeLU (exact tanh approx, with backward)
// ─────────────────────────────────────────────
struct GeLU {
    mutable vector<float> xc;
    static constexpr float C=0.7978845608f;
    void fwd(const vector<float>& x,vector<float>& o) const {
        xc=x; o.resize(x.size());
        for(int i=0;i<(int)x.size();i++){
            float v=x[i], t=tanhf(C*(v+0.044715f*v*v*v));
            o[i]=0.5f*v*(1+t);
        }
    }
    void bwd(const vector<float>& do_,vector<float>& dx){
        dx.resize(xc.size());
        for(int i=0;i<(int)xc.size();i++){
            float v=xc[i], t=tanhf(C*(v+0.044715f*v*v*v));
            float s2=1-t*t, dt=C*(1+3*0.044715f*v*v);
            dx[i]=do_[i]*(0.5f*(1+t)+0.5f*v*s2*dt);
        }
    }
};

// ─────────────────────────────────────────────
// Dropout
// ─────────────────────────────────────────────
struct Dropout {
    float p; mutable vector<float> mask;
    Dropout(float p_=0.f):p(p_){}
    void fwd(const vector<float>& x,vector<float>& o) const {
        o.resize(x.size()); mask.resize(x.size());
        if(!TRAINING||p==0.f){o=x;fill(mask.begin(),mask.end(),1.f);return;}
        float sc=1.f/(1.f-p);
        for(int i=0;i<(int)x.size();i++){mask[i]=flip(1-p)?sc:0.f;o[i]=x[i]*mask[i];}
    }
    void bwd(const vector<float>& do_,vector<float>& dx){
        dx.resize(do_.size()); for(int i=0;i<(int)do_.size();i++) dx[i]=do_[i]*mask[i];
    }
};

// ─────────────────────────────────────────────
// KV Cache  (for inference)
// ─────────────────────────────────────────────
struct KVCache {
    int n_head, head_dim;
    vector<vector<float>> K, V;   // [head][seq * head_dim]
    int seq_len=0;

    KVCache()=default;
    KVCache(int nh,int hd):n_head(nh),head_dim(hd),K(nh),V(nh){}
    void clear(){ for(auto& k:K) k.clear(); for(auto& v:V) v.clear(); seq_len=0; }

    void append(int h,const float* k,const float* v){
        K[h].insert(K[h].end(),k,k+head_dim);
        V[h].insert(V[h].end(),v,v+head_dim);
    }
};

// ─────────────────────────────────────────────
// Causal Self-Attention (full backward + KV cache inference)
// ─────────────────────────────────────────────
struct CausalSelfAttn {
    int ne,nh,hd;
    Linear c_attn, c_proj;
    Dropout attn_dp, resid_dp;
    mutable int Tc=0;
    mutable vector<float> qkv_c, q_c, k_c, v_c, aw_c, ctx_c;

    CausalSelfAttn()=default;
    CausalSelfAttn(int n_embd,int n_head,float dp=0.f)
        :ne(n_embd),nh(n_head),hd(n_embd/n_head),
         c_attn(n_embd,3*n_embd,true,0.02f),
         c_proj(n_embd,n_embd,true,0.02f/sqrtf(2.f)),
         attn_dp(dp),resid_dp(dp){}

    void fwd(const vector<float>& x,int T,vector<float>& out,KVCache* kvc=nullptr) const {
        Tc=T;
        c_attn.fwd2d(x,T,qkv_c);
        q_c.resize(nh*T*hd); k_c.resize(nh*T*hd); v_c.resize(nh*T*hd);
        for(int t=0;t<T;t++) for(int h=0;h<nh;h++) for(int d=0;d<hd;d++){
            q_c[h*T*hd+t*hd+d]=qkv_c[t*3*ne+h*hd+d];
            k_c[h*T*hd+t*hd+d]=qkv_c[t*3*ne+ne+h*hd+d];
            v_c[h*T*hd+t*hd+d]=qkv_c[t*3*ne+2*ne+h*hd+d];
        }
        float sc=1.f/sqrtf((float)hd);
        aw_c.assign(nh*T*T,0.f); ctx_c.assign(T*ne,0.f);
        for(int h=0;h<nh;h++){
            float* aw=aw_c.data()+h*T*T;
            const float* qh=q_c.data()+h*T*hd, *kh=k_c.data()+h*T*hd, *vh=v_c.data()+h*T*hd;
            for(int i=0;i<T;i++){
                float mx=-1e30f;
                for(int j=0;j<=i;j++){
                    float s=0; for(int d=0;d<hd;d++) s+=qh[i*hd+d]*kh[j*hd+d];
                    aw[i*T+j]=s*sc; mx=max(mx,aw[i*T+j]);
                }
                for(int j=i+1;j<T;j++) aw[i*T+j]=-1e30f;
                float sm=0; for(int j=0;j<T;j++){aw[i*T+j]=safe_exp(aw[i*T+j]-mx);sm+=aw[i*T+j];}
                for(int j=0;j<T;j++) aw[i*T+j]/=sm;
            }
            vector<float> aw_src(aw,aw+T*T),aw_dp;
            attn_dp.fwd(aw_src,aw_dp);
            copy(aw_dp.begin(),aw_dp.end(),aw);
            for(int i=0;i<T;i++) for(int d=0;d<hd;d++){
                float s=0; for(int j=0;j<T;j++) s+=aw[i*T+j]*vh[j*hd+d];
                ctx_c[i*ne+h*hd+d]=s;
            }
        }
        vector<float> proj,dp_out;
        c_proj.fwd2d(ctx_c,T,proj);
        resid_dp.fwd(proj,dp_out);
        out=dp_out;
    }

    void bwd(const vector<float>& dout_d,vector<float>& dx){
        int T=Tc;
        vector<float> dout; resid_dp.bwd(dout_d,dout);
        vector<float> dctx; c_proj.bwd2d(dout,T,dctx);
        vector<float> dq(nh*T*hd,0),dk(nh*T*hd,0),dv(nh*T*hd,0);
        for(int h=0;h<nh;h++){
            const float* aw=aw_c.data()+h*T*T, *vh=v_c.data()+h*T*hd;
            float* dvh=dv.data()+h*T*hd;
            vector<float> daw_raw(T*T,0.f);
            for(int i=0;i<T;i++) for(int d=0;d<hd;d++){
                float g=dctx[i*ne+h*hd+d];
                for(int j=0;j<T;j++){daw_raw[i*T+j]+=g*vh[j*hd+d];dvh[j*hd+d]+=g*aw[i*T+j];}
            }
            vector<float> daw_pre; attn_dp.bwd(daw_raw,daw_pre);
            const float* qh=q_c.data()+h*T*hd, *kh=k_c.data()+h*T*hd;
            float* dqh=dq.data()+h*T*hd, *dkh=dk.data()+h*T*hd;
            for(int i=0;i<T;i++){
                const float* awi=aw+i*T; float* dp=daw_pre.data()+i*T;
                float dot=0; for(int j=0;j<=i;j++) dot+=dp[j]*awi[j];
                for(int j=0;j<=i;j++) dp[j]=awi[j]*(dp[j]-dot);
                for(int j=i+1;j<T;j++) dp[j]=0;
                float isc=1.f/sqrtf((float)hd);
                for(int j=0;j<=i;j++) for(int d=0;d<hd;d++){
                    dqh[i*hd+d]+=dp[j]*kh[j*hd+d]*isc;
                    dkh[j*hd+d]+=dp[j]*qh[i*hd+d]*isc;
                }
            }
        }
        vector<float> dqkv(T*3*ne,0.f);
        for(int t=0;t<T;t++) for(int h=0;h<nh;h++) for(int d=0;d<hd;d++){
            dqkv[t*3*ne+h*hd+d]       +=dq[h*T*hd+t*hd+d];
            dqkv[t*3*ne+ne+h*hd+d]    +=dk[h*T*hd+t*hd+d];
            dqkv[t*3*ne+2*ne+h*hd+d]  +=dv[h*T*hd+t*hd+d];
        }
        c_attn.bwd2d(dqkv,T,dx);
    }

    void reg(AdamW& opt){ c_attn.reg(opt); c_proj.reg(opt); }
    void save(ofstream& f) const { c_attn.save(f); c_proj.save(f); }
    void load(ifstream& f){ c_attn.load(f); c_proj.load(f); }
};

// ─────────────────────────────────────────────
// MLP block
// ─────────────────────────────────────────────
struct MLP {
    int ne;
    Linear fc1, fc2; GeLU gelu; Dropout dp;
    mutable vector<float> h1pre,h1;

    MLP()=default;
    MLP(int n_embd,float dropout=0.f)
        :ne(n_embd),fc1(n_embd,4*n_embd,true,0.02f),
         fc2(4*n_embd,n_embd,true,0.02f/sqrtf(2.f)),dp(dropout){}

    void fwd(const vector<float>& x,int T,vector<float>& o) const {
        fc1.fwd2d(x,T,h1pre); gelu.fwd(h1pre,h1);
        vector<float> h2; fc2.fwd2d(h1,T,h2); dp.fwd(h2,o);
    }
    void bwd(const vector<float>& do_d,vector<float>& dx){
        int T=h1.size()/(4*ne);
        vector<float> do_,dh1,dh1pre;
        dp.bwd(do_d,do_); fc2.bwd2d(do_,T,dh1); gelu.bwd(dh1,dh1pre); fc1.bwd2d(dh1pre,T,dx);
    }
    void reg(AdamW& opt){ fc1.reg(opt); fc2.reg(opt); }
    void save(ofstream& f) const { fc1.save(f); fc2.save(f); }
    void load(ifstream& f){ fc1.load(f); fc2.load(f); }
};

// ─────────────────────────────────────────────
// Transformer block
// ─────────────────────────────────────────────
struct Block {
    int ne;
    LayerNorm ln1, ln2;
    CausalSelfAttn attn;
    MLP mlp;
    mutable vector<float> xin, ln1o, x2, ln2o;
    mutable int Tc=0;

    Block()=default;
    Block(int n_embd,int n_head,float dp=0.f)
        :ne(n_embd),ln1(n_embd),ln2(n_embd),attn(n_embd,n_head,dp),mlp(n_embd,dp){}

    void fwd(const vector<float>& x,int T,vector<float>& o) const {
        Tc=T; xin=x;
        ln1.fwd(x,T,ln1o);
        vector<float> ar; attn.fwd(ln1o,T,ar);
        x2.resize(T*ne); for(int i=0;i<T*ne;i++) x2[i]=x[i]+ar[i];
        ln2.fwd(x2,T,ln2o);
        vector<float> mr; mlp.fwd(ln2o,T,mr);
        o.resize(T*ne); for(int i=0;i<T*ne;i++) o[i]=x2[i]+mr[i];
    }

    void bwd(const vector<float>& dout,vector<float>& dx){
        int T=Tc;
        vector<float> dln2,dmr; mlp.bwd(dout,dln2); ln2.bwd(dln2,dmr);
        vector<float> dx2(T*ne); for(int i=0;i<T*ne;i++) dx2[i]=dout[i]+dmr[i];
        vector<float> dattn,dln1; attn.bwd(dx2,dattn); ln1.bwd(dattn,dln1);
        dx.resize(T*ne); for(int i=0;i<T*ne;i++) dx[i]=dx2[i]+dln1[i];
    }

    void reg(AdamW& opt){ ln1.reg(opt); ln2.reg(opt); attn.reg(opt); mlp.reg(opt); }
    void save(ofstream& f) const { ln1.save(f); ln2.save(f); attn.save(f); mlp.save(f); }
    void load(ifstream& f){ ln1.load(f); ln2.load(f); attn.load(f); mlp.load(f); }
};

// ─────────────────────────────────────────────
// BPE Tokenizer
// ─────────────────────────────────────────────
struct BPETokenizer {
    unordered_map<string,int> vocab;
    vector<string> id_to_tok;
    vector<pair<int,int>> merges_order;
    map<pair<int,int>,int> merges;
    int vocab_size=0;

    static constexpr int UNK=0, BOS=1, EOS=2, PAD=3;

    void build(const string& text, int target_vocab=512, int min_freq=2){
        unordered_map<string,int> word_freq;
        istringstream ss(text);
        string word;
        while(ss>>word) word_freq[word]++;

        map<vector<int>,int> corpus;
        for(auto& [w,f]:word_freq){
            vector<int> chars;
            for(unsigned char c:w) chars.push_back(c+4);
            corpus[chars]=f;
        }

        id_to_tok={"<unk>","<bos>","<eos>","<pad>"};
        for(int i=0;i<256;i++){
            string s; s+=(char)i;
            id_to_tok.push_back(s);
            vocab[s]=i+4;
        }
        vocab_size=4+256;

        while(vocab_size<target_vocab){
            map<pair<int,int>,int> pairs;
            for(auto& [seq,f]:corpus){
                for(int i=0;i<(int)seq.size()-1;i++)
                    pairs[{seq[i],seq[i+1]}]+=f;
            }
            if(pairs.empty()) break;
            auto best=max_element(pairs.begin(),pairs.end(),
                [](auto& a,auto& b){return a.second<b.second;});
            if(best->second<min_freq) break;

            auto [p0,p1]=best->first;
            int new_id=vocab_size++;
            string new_tok=id_to_tok[p0]+id_to_tok[p1];
            id_to_tok.push_back(new_tok);
            vocab[new_tok]=new_id;
            merges[{p0,p1}]=new_id;
            merges_order.push_back({p0,p1});

            map<vector<int>,int> new_corpus;
            for(auto& [seq,f]:corpus){
                vector<int> ns;
                for(int i=0;i<(int)seq.size();){
                    if(i<(int)seq.size()-1&&seq[i]==p0&&seq[i+1]==p1){
                        ns.push_back(new_id); i+=2;
                    } else { ns.push_back(seq[i]); i++; }
                }
                new_corpus[ns]+=f;
            }
            corpus=new_corpus;
        }

        for(int i=0;i<(int)id_to_tok.size();i++) vocab[id_to_tok[i]]=i;
        vocab_size=id_to_tok.size();
    }

    vector<int> encode(const string& text) const {
        vector<int> out={BOS};
        istringstream ss(text); string word;
        while(ss>>word){
            vector<int> seq;
            for(unsigned char c:word) seq.push_back(c+4);
            bool changed=true;
            while(changed){
                changed=false;
                for(int i=0;i<(int)seq.size()-1;i++){
                    auto it=merges.find({seq[i],seq[i+1]});
                    if(it!=merges.end()){
                        int nid=it->second;
                        vector<int> ns(seq.begin(),seq.begin()+i);
                        ns.push_back(nid);
                        ns.insert(ns.end(),seq.begin()+i+2,seq.end());
                        seq=ns; changed=true; break;
                    }
                }
            }
            for(int t:seq) out.push_back(t);
            out.push_back(4+32);  // space byte
        }
        out.push_back(EOS);
        return out;
    }

    string decode(const vector<int>& ids) const {
        string out;
        for(int id:ids){
            if(id==BOS||id==EOS||id==PAD) continue;
            if(id<0||id>=(int)id_to_tok.size()) out+="?";
            else out+=id_to_tok[id];
        }
        return out;
    }

    void save(ofstream& f) const {
        f.write((char*)&vocab_size,4);
        int nm=merges_order.size(); f.write((char*)&nm,4);
        for(auto& [a,b]:merges_order){ f.write((char*)&a,4); f.write((char*)&b,4); }
        for(auto& s:id_to_tok){
            int len=s.size(); f.write((char*)&len,4); f.write(s.data(),len);
        }
    }

    void load(ifstream& f){
        f.read((char*)&vocab_size,4);
        int nm; f.read((char*)&nm,4);
        merges.clear(); merges_order.clear();
        id_to_tok.assign(vocab_size,"");
        for(int i=0;i<nm;i++){
            int a,b; f.read((char*)&a,4); f.read((char*)&b,4);
            merges_order.push_back({a,b});
        }
        int cur_id=4+256;
        for(int i=0;i<nm;i++) merges[merges_order[i]]=cur_id++;
        for(int i=0;i<vocab_size;i++){
            int len; f.read((char*)&len,4);
            id_to_tok[i].resize(len); f.read(id_to_tok[i].data(),len);
            vocab[id_to_tok[i]]=i;
        }
    }
};

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────
struct Config {
    int vocab_size=512, block_size=128, n_embd=128, n_head=4, n_layer=4;
    float dropout=0.1f;
    int batch_size=4;
    float lr=3e-4f, lr_min=3e-5f, weight_decay=0.1f, grad_clip=1.f;
    int warmup=200, decay_steps=10000;
    int eval_every=100, save_every=1000;
    int lora_rank=0;       // 0 = no lora
    float lora_alpha=1.f;
    bool use_bpe=true;
    int bpe_vocab=512;
};

// ─────────────────────────────────────────────
// Vibes model
// ─────────────────────────────────────────────
struct Vibes {
    Config cfg;
    Param wte, wpe;
    vector<Block> blocks;
    LayerNorm ln_f;
    Linear lm_head;

    Vibes()=default;
    Vibes(Config c):cfg(c),
        wte(c.vocab_size,c.n_embd,0.02f),
        wpe(c.block_size,c.n_embd,0.01f),
        ln_f(c.n_embd),
        lm_head(c.n_embd,c.vocab_size,false,0.02f){
        for(int i=0;i<c.n_layer;i++) blocks.emplace_back(c.n_embd,c.n_head,c.dropout);
        // weight tying
        lm_head.W.w=wte.w;
    }

    void attach_lora(int rank,float alpha){
        for(auto& b:blocks){
            b.attn.c_attn.attach_lora(rank,alpha);
            b.attn.c_proj.attach_lora(rank,alpha);
        }
    }

    // forward + optional backward, returns loss per token
    float fwd_bwd(const vector<int>& idx,const vector<int>& tgt,bool do_bwd){
        int T=idx.size();
        assert(T<=(int)cfg.block_size);

        vector<float> x(T*cfg.n_embd);
        for(int t=0;t<T;t++) for(int i=0;i<cfg.n_embd;i++)
            x[t*cfg.n_embd+i]=wte.W(idx[t],i)+wpe.W(t,i);

        vector<vector<float>> acts={x};
        for(auto& blk:blocks){ vector<float> o; blk.fwd(acts.back(),T,o); acts.push_back(o); }
        vector<float> lno; ln_f.fwd(acts.back(),T,lno);

        float loss=0.f;
        vector<float> dlno(T*cfg.n_embd,0.f);
        for(int t=0;t<T;t++){
            vector<float> lt(lno.begin()+t*cfg.n_embd,lno.begin()+(t+1)*cfg.n_embd);
            vector<float> logits; lm_head.fwd1d(lt,logits);
            float mx=*max_element(logits.begin(),logits.end());
            float sm=0; for(float v:logits) sm+=safe_exp(v-mx);
            loss+=-(logits[tgt[t]]-mx-safe_log(sm));
            if(do_bwd){
                vector<float> pr(cfg.vocab_size);
                for(int i=0;i<cfg.vocab_size;i++) pr[i]=safe_exp(logits[i]-mx)/sm;
                pr[tgt[t]]-=1.f;
                vector<float> dlt; lm_head.bwd1d(pr,dlt);
                for(int i=0;i<cfg.n_embd;i++) dlno[t*cfg.n_embd+i]=dlt[i];
            }
        }
        loss/=T;
        if(!do_bwd) return loss;
        for(auto& v:dlno) v/=T;

        vector<float> dx; ln_f.bwd(dlno,dx);
        for(int li=(int)blocks.size()-1;li>=0;li--){ vector<float> dxp; blocks[li].bwd(dx,dxp); dx=dxp; }
        for(int t=0;t<T;t++) for(int i=0;i<cfg.n_embd;i++){
            float g=dx[t*cfg.n_embd+i];
            wte.G(idx[t],i)+=g; wpe.G(t,i)+=g;
        }
        // sync weight-tied lm_head gradient
        for(int i=0;i<cfg.vocab_size*cfg.n_embd;i++) wte.g[i]+=lm_head.W.g[i];
        return loss;
    }

    // get logits for last token in context
    void get_logits(const vector<int>& ctx,vector<float>& logits) const {
        int T=ctx.size();
        vector<float> x(T*cfg.n_embd);
        for(int t=0;t<T;t++) for(int i=0;i<cfg.n_embd;i++)
            x[t*cfg.n_embd+i]=wte.W(ctx[t],i)+wpe.W(t,i);
        vector<float> cur=x;
        for(auto& blk:blocks){ vector<float> o; blk.fwd(cur,T,o); cur=o; }
        vector<float> lno; ln_f.fwd(cur,T,lno);
        vector<float> last(lno.end()-cfg.n_embd,lno.end());
        lm_head.fwd1d(last,logits);
    }

    void zero_grad(){
        wte.zero_grad(); wpe.zero_grad(); lm_head.W.zero_grad();
        ln_f.gamma.zero_grad(); ln_f.beta.zero_grad();
        for(auto& b:blocks){
            b.ln1.gamma.zero_grad(); b.ln1.beta.zero_grad();
            b.attn.c_attn.W.zero_grad(); b.attn.c_attn.b.zero_grad();
            b.attn.c_proj.W.zero_grad(); b.attn.c_proj.b.zero_grad();
            b.attn.c_attn.lora.zero_grad(); b.attn.c_proj.lora.zero_grad();
            b.ln2.gamma.zero_grad(); b.ln2.beta.zero_grad();
            b.mlp.fc1.W.zero_grad(); b.mlp.fc1.b.zero_grad();
            b.mlp.fc2.W.zero_grad(); b.mlp.fc2.b.zero_grad();
        }
    }

    void reg(AdamW& opt, bool lora_only=false){
        if(!lora_only){
            opt.add(&wte); opt.add(&wpe);
            for(auto& b:blocks){ b.reg(opt); }
            ln_f.reg(opt); opt.add(&lm_head.W);
        } else {
            for(auto& b:blocks){
                b.attn.c_attn.reg_lora(opt);
                b.attn.c_proj.reg_lora(opt);
            }
        }
    }

    int n_params() const {
        int n=wte.numel()+wpe.numel()+ln_f.gamma.numel()+ln_f.beta.numel()+lm_head.W.numel();
        for(auto& b:blocks){
            n+=b.ln1.gamma.numel()+b.ln1.beta.numel()+b.ln2.gamma.numel()+b.ln2.beta.numel();
            n+=b.attn.c_attn.W.numel()+b.attn.c_attn.b.numel();
            n+=b.attn.c_proj.W.numel()+b.attn.c_proj.b.numel();
            n+=b.mlp.fc1.W.numel()+b.mlp.fc1.b.numel();
            n+=b.mlp.fc2.W.numel()+b.mlp.fc2.b.numel();
        }
        return n;
    }

    void save_ckpt(const string& path,AdamW& opt) const {
        ofstream f(path,ios::binary); if(!f){cerr<<"save fail\n";return;}
        f.write((char*)&cfg,sizeof(cfg));
        wte.save(f); wpe.save(f);
        for(auto& b:blocks) b.save(f);
        ln_f.save(f); lm_head.save(f);
        opt.save(f);
        cerr<<"[vibes] ckpt -> "<<path<<"\n";
    }

    bool load_ckpt(const string& path,AdamW& opt){
        ifstream f(path,ios::binary); if(!f) return false;
        f.read((char*)&cfg,sizeof(cfg));
        wte.load(f); wpe.load(f);
        for(auto& b:blocks) b.load(f);
        ln_f.load(f); lm_head.load(f);
        opt.load(f);
        cerr<<"[vibes] ckpt <- "<<path<<"\n";
        return true;
    }
};

// ─────────────────────────────────────────────
// Sampling strategies
// ─────────────────────────────────────────────

// apply temperature + top-k + top-p, sample
int sample_token(const vector<float>& logits,float temp,int top_k,float top_p,
                 const vector<int>& generated,float rep_pen=1.1f){
    int V=logits.size();
    vector<float> lg=logits;

    // repetition penalty
    if(rep_pen!=1.f){
        set<int> seen(generated.end()-min((int)generated.size(),64),generated.end());
        for(int id:seen) lg[id]=(lg[id]>0)?lg[id]/rep_pen:lg[id]*rep_pen;
    }

    // temperature
    for(auto& v:lg) v/=max(temp,1e-6f);

    // top-k
    if(top_k>0&&top_k<V){
        vector<int> idx(V); iota(idx.begin(),idx.end(),0);
        partial_sort(idx.begin(),idx.begin()+top_k,idx.end(),[&](int a,int b){return lg[a]>lg[b];});
        float thresh=lg[idx[top_k-1]];
        for(int i=0;i<V;i++) if(lg[i]<thresh) lg[i]=-1e30f;
    }

    // softmax
    float mx=*max_element(lg.begin(),lg.end());
    vector<float> pr(V); float sm=0;
    for(int i=0;i<V;i++){pr[i]=safe_exp(lg[i]-mx);sm+=pr[i];}
    for(auto& p:pr) p/=sm;

    // top-p nucleus
    if(top_p>0&&top_p<1.f){
        vector<int> ord(V); iota(ord.begin(),ord.end(),0);
        sort(ord.begin(),ord.end(),[&](int a,int b){return pr[a]>pr[b];});
        float cum=0; int cutoff=V-1;
        for(int k=0;k<V;k++){cum+=pr[ord[k]];if(cum>=top_p){cutoff=k;break;}}
        for(int k=cutoff+1;k<V;k++) pr[ord[k]]=0;
        float s2=0; for(float p:pr) s2+=p; for(auto& p:pr) p/=s2;
    }

    float r=randf(0,1),cum=0;
    for(int i=0;i<V;i++){cum+=pr[i];if(r<=cum) return i;}
    return V-1;
}

// beam search
vector<int> beam_search(Vibes& model,vector<int> prefix,int max_new,
                         int beam_width,float temp,int eos_id){
    struct Beam{ vector<int> toks; float score; };
    vector<Beam> beams={{prefix,0.f}};

    for(int step=0;step<max_new;step++){
        vector<pair<float,Beam>> candidates;
        for(auto& beam:beams){
            int ct=min((int)beam.toks.size(),(int)model.cfg.block_size);
            vector<int> ctx(beam.toks.end()-ct,beam.toks.end());
            vector<float> logits; model.get_logits(ctx,logits);
            float mx=*max_element(logits.begin(),logits.end());
            float sm=0; for(float v:logits) sm+=safe_exp(v-mx);
            vector<int> topk_ids(logits.size()); iota(topk_ids.begin(),topk_ids.end(),0);
            partial_sort(topk_ids.begin(),topk_ids.begin()+min(beam_width,(int)logits.size()),
                topk_ids.end(),[&](int a,int b){return logits[a]>logits[b];});
            for(int k=0;k<beam_width;k++){
                int id=topk_ids[k];
                float lp=logits[id]-mx-safe_log(sm);
                Beam nb=beam; nb.toks.push_back(id); nb.score+=lp;
                candidates.push_back({nb.score/(nb.toks.size()),nb});
            }
        }
        sort(candidates.begin(),candidates.end(),[](auto& a,auto& b){return a.first>b.first;});
        beams.clear();
        for(int i=0;i<min(beam_width,(int)candidates.size());i++) beams.push_back(candidates[i].second);
        if(beams[0].toks.back()==eos_id) break;
    }
    return beams[0].toks;
}

// ─────────────────────────────────────────────
// Cosine LR schedule
// ─────────────────────────────────────────────
float cosine_lr(float base,float mn,int step,int warm,int decay){
    if(step<warm) return base*(float)(step+1)/max(1,warm);
    if(step>=decay) return mn;
    float p=(float)(step-warm)/(decay-warm);
    return mn+0.5f*(base-mn)*(1+cosf(M_PI*p));
}

// ─────────────────────────────────────────────
// Training loop
// ─────────────────────────────────────────────
void train_loop(Vibes& model,const vector<int>& train_ids,const vector<int>& val_ids,
                Config& cfg,AdamW& opt,const string& ckpt_path){
    int TN=train_ids.size(), VS=val_ids.size();
    uniform_int_distribution<int> dist(0,TN-cfg.block_size-2);
    float smooth=-1.f;
    auto t0=clk::now();
    cerr<<fixed<<setprecision(4);
    cerr<<"[vibes] params:"<<model.n_params()<<" train:"<<TN<<" val:"<<VS<<"\n";

    for(;;){
        TRAINING=true;
        model.zero_grad();
        float bl=0;
        for(int bi=0;bi<cfg.batch_size;bi++){
            int s=dist(RNG);
            vector<int> inp(train_ids.begin()+s, train_ids.begin()+s+cfg.block_size);
            vector<int> tgt(train_ids.begin()+s+1,train_ids.begin()+s+cfg.block_size+1);
            bl+=model.fwd_bwd(inp,tgt,true);
        }
        bl/=cfg.batch_size;
        for(auto* p:opt.ps) for(auto& g:p->g) g/=cfg.batch_size;
        float lrs=cosine_lr(1.f,cfg.lr_min/cfg.lr,opt.t,cfg.warmup,cfg.decay_steps);
        opt.step(lrs);
        smooth=(smooth<0)?bl:0.9f*smooth+0.1f*bl;

        if(opt.t%cfg.eval_every==0){
            TRAINING=false;
            float vl=0; int vs=0;
            for(int s=0;s<VS-cfg.block_size-1&&vs<8;s+=cfg.block_size,vs++){
                vector<int> vi(val_ids.begin()+s,val_ids.begin()+s+cfg.block_size);
                vector<int> vt(val_ids.begin()+s+1,val_ids.begin()+s+cfg.block_size+1);
                vl+=model.fwd_bwd(vi,vt,false);
            }
            float secs=chrono::duration<float>(clk::now()-t0).count();
            float cur_lr=cfg.lr*lrs;
            cerr<<"step "<<setw(6)<<opt.t
                <<" | train "<<smooth
                <<" | val "<<(vs?vl/vs:0.f)
                <<" | lr "<<scientific<<cur_lr<<fixed
                <<" | "<<setprecision(1)<<secs<<"s\n"<<setprecision(4);
        }
        if(opt.t%cfg.save_every==0) model.save_ckpt(ckpt_path,opt);
    }
}

// ─────────────────────────────────────────────
// Personality  — baked into generation context,
// never trained in. swappable at runtime.
// ─────────────────────────────────────────────
struct Personality {
    string name;
    string system_prompt;
    float temperature;
    float rep_penalty;

    static Personality aight_bet(){
        return {
            "aight_bet",
            "<|system|>vibes gpt. no filler. no grammar corrections. "
            "embraces memes, typos, lowercase energy. "
            "never says 'certainly' or 'of course' or 'great question'. "
            "talks like: aight bet, no cap, fr fr, lowkey, bussin, ngl. "
            "keeps it real. short answers hit different. vibe check passed.<|endoftext|>",
            0.85f,
            1.15f
        };
    }

    static Personality professional(){
        return {"professional","",0.7f,1.1f};
    }

    static Personality chaotic(){
        return {
            "chaotic",
            "<|system|>unhinged. unpredictable. maximum entropy. chaos mode engaged.<|endoftext|>",
            1.2f,
            1.0f
        };
    }
};

// ─────────────────────────────────────────────
// main
// ─────────────────────────────────────────────
void print_usage(const char* n){
    cerr<<"usage:\n"
        <<"  "<<n<<" train --data <f> [--ckpt <f>] [--tok <f>] [--resume]\n"
        <<"             [--n_embd N] [--n_head N] [--n_layer N] [--block N]\n"
        <<"             [--batch N] [--lr F] [--dropout F] [--warmup N] [--decay N]\n"
        <<"             [--eval N] [--save N] [--bpe_vocab N] [--lora_rank N]\n"
        <<"  "<<n<<" gen  --ckpt <f> --tok <f> [--prompt <s>] [--beam N]\n"
        <<"             [--max_new N] [--temp F] [--top_k N] [--top_p F]\n"
        <<"             [--rep_pen F] [--personality aight_bet|professional|chaotic]\n";
}

int main(int argc,char** argv){
    if(argc<2){ print_usage(argv[0]); return 1; }
    string mode=argv[1];

    Config cfg;
    string data_path, ckpt_path="vibes.ckpt", tok_path="vibes.tok", prompt="";
    string personality_name="aight_bet";
    bool resume=false;
    int max_new=200, beam_width=1, top_k=40;
    float temp=-1.f, top_p=0.9f, rep_pen=-1.f;

    for(int i=2;i<argc;i++){
        string a=argv[i];
        auto nx=[&]()->string{ return (i+1<argc)?string(argv[++i]):""; };
        if(a=="--data") data_path=nx();
        else if(a=="--ckpt") ckpt_path=nx();
        else if(a=="--tok") tok_path=nx();
        else if(a=="--prompt") prompt=nx();
        else if(a=="--resume") resume=true;
        else if(a=="--n_embd") cfg.n_embd=stoi(nx());
        else if(a=="--n_head") cfg.n_head=stoi(nx());
        else if(a=="--n_layer") cfg.n_layer=stoi(nx());
        else if(a=="--block") cfg.block_size=stoi(nx());
        else if(a=="--batch") cfg.batch_size=stoi(nx());
        else if(a=="--lr") cfg.lr=stof(nx());
        else if(a=="--dropout") cfg.dropout=stof(nx());
        else if(a=="--warmup") cfg.warmup=stoi(nx());
        else if(a=="--decay") cfg.decay_steps=stoi(nx());
        else if(a=="--eval") cfg.eval_every=stoi(nx());
        else if(a=="--save") cfg.save_every=stoi(nx());
        else if(a=="--bpe_vocab") cfg.bpe_vocab=stoi(nx());
        else if(a=="--lora_rank") cfg.lora_rank=stoi(nx());
        else if(a=="--max_new") max_new=stoi(nx());
        else if(a=="--temp") temp=stof(nx());
        else if(a=="--top_k") top_k=stoi(nx());
        else if(a=="--top_p") top_p=stof(nx());
        else if(a=="--rep_pen") rep_pen=stof(nx());
        else if(a=="--beam") beam_width=stoi(nx());
        else if(a=="--personality") personality_name=nx();
        else { cerr<<"unknown: "<<a<<"\n"; return 1; }
    }
    cfg.n_embd=(cfg.n_embd/cfg.n_head)*cfg.n_head;

    // pick personality
    Personality persona;
    if(personality_name=="professional") persona=Personality::professional();
    else if(personality_name=="chaotic") persona=Personality::chaotic();
    else persona=Personality::aight_bet();
    if(temp<0) temp=persona.temperature;
    if(rep_pen<0) rep_pen=persona.rep_penalty;

    if(mode=="train"){
        if(data_path.empty()){ cerr<<"need --data\n"; return 1; }
        ifstream f(data_path); if(!f){ cerr<<"cant open "<<data_path<<"\n"; return 1; }
        string text((istreambuf_iterator<char>(f)),istreambuf_iterator<char>());

        BPETokenizer tok;
        cerr<<"[vibes] building BPE vocab (target="<<cfg.bpe_vocab<<")...\n";
        tok.build(text,cfg.bpe_vocab);
        cfg.vocab_size=tok.vocab_size;
        cerr<<"[vibes] vocab size: "<<tok.vocab_size<<"\n";

        ofstream tf(tok_path,ios::binary); tok.save(tf);
        cerr<<"[vibes] tok -> "<<tok_path<<"\n";

        auto all_ids=tok.encode(text);
        int N=all_ids.size();
        int vsz=max(cfg.block_size+2,(int)(N*0.1f));
        vector<int> val_ids(all_ids.begin(),all_ids.begin()+vsz);
        vector<int> train_ids(all_ids.begin()+vsz,all_ids.end());

        cerr<<"[vibes] embd:"<<cfg.n_embd<<" heads:"<<cfg.n_head
            <<" layers:"<<cfg.n_layer<<" block:"<<cfg.block_size
            <<" batch:"<<cfg.batch_size<<" lora_rank:"<<cfg.lora_rank<<"\n";

        Vibes model(cfg);
        if(cfg.lora_rank>0) model.attach_lora(cfg.lora_rank,cfg.lora_alpha);

        AdamW opt(cfg.lr,0.9f,0.999f,1e-8f,cfg.weight_decay,cfg.grad_clip);
        model.reg(opt, cfg.lora_rank>0);

        if(resume) model.load_ckpt(ckpt_path,opt);
        train_loop(model,train_ids,val_ids,cfg,opt,ckpt_path);

    } else if(mode=="gen"){
        ifstream tf(tok_path,ios::binary);
        if(!tf){ cerr<<"need --tok (run train first)\n"; return 1; }
        BPETokenizer tok; tok.load(tf);
        cfg.vocab_size=tok.vocab_size;

        Vibes model(cfg);
        if(cfg.lora_rank>0) model.attach_lora(cfg.lora_rank,cfg.lora_alpha);
        AdamW opt(cfg.lr); model.reg(opt, cfg.lora_rank>0);
        if(!model.load_ckpt(ckpt_path,opt)){ cerr<<"no ckpt: "<<ckpt_path<<"\n"; return 1; }

        TRAINING=false;

        // prepend personality system prompt
        string full_input=persona.system_prompt+(prompt.empty()?"":prompt);
        auto enc=tok.encode(full_input);
        if(enc.empty()) enc={BPETokenizer::BOS};
        int ctx_len=min((int)enc.size(),cfg.block_size);
        vector<int> ctx(enc.end()-ctx_len,enc.end());

        if(beam_width>1){
            auto out=beam_search(model,ctx,max_new,beam_width,temp,BPETokenizer::EOS);
            cout<<tok.decode(vector<int>(out.begin()+ctx.size(),out.end()))<<"\n";
        } else {
            vector<int> generated=ctx;
            for(int s=0;s<max_new;s++){
                int ct=min((int)generated.size(),cfg.block_size);
                vector<int> window(generated.end()-ct,generated.end());
                vector<float> logits; model.get_logits(window,logits);
                int next=sample_token(logits,temp,top_k,top_p,generated,rep_pen);
                if(next==BPETokenizer::EOS) break;
                generated.push_back(next);
            }
            cout<<tok.decode(vector<int>(generated.begin()+ctx.size(),generated.end()))<<"\n";
        }

    } else { print_usage(argv[0]); return 1; }
    return 0;
}
