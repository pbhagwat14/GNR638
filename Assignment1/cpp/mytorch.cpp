#include "mytorch.h"

#include <random>
#include <cmath>
#include <unordered_set>

/* ===== Tensor ===== */

Tensor::Tensor(std::vector<float> values, std::vector<int> shp, bool req)
    : data(values), grad(values.size(), 0.0f), shape(shp), requires_grad(req) {
            parents.clear();
            backward_fn = nullptr;
            // make sure grad length matches data
            grad.assign(data.size(), 0.0f);

    }
void Tensor::backward() {
    if (!requires_grad) return;

    // Step 1: initialize loss gradient
    bool all_zero = true;
    for(float g : grad)
        if(g != 0.0f) { all_zero=false; break; }

    if(all_zero)
        for(float &g:grad) g=1.0f;

    // Step 2: build topological order using post-order DFS
    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;
    
    std::function<void(std::shared_ptr<Tensor>)> build_topo = 
        [&](std::shared_ptr<Tensor> node) {
            if(visited.count(node.get())) return;
            visited.insert(node.get());
            
            // Visit parents first (recursive DFS)
            for(auto &parent : node->parents)
                build_topo(parent);
            
            // Add current node after visiting all parents
            topo.push_back(node);
        };
    
    build_topo(shared_from_this());

    // Step 3: run backward in REVERSE topological order
    for(auto it = topo.rbegin(); it != topo.rend(); ++it){
        if((*it)->backward_fn)
            (*it)->backward_fn();
    }
}

/* ===== SGD Optimizer ===== */

SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters, float learning_rate)
    : params(parameters), lr(learning_rate) {}

void SGD::step() {
    for(auto &p : params){
        if(!p->requires_grad) continue;

        for(size_t i=0;i<p->data.size();i++)
            p->data[i] -= lr * p->grad[i];
    }
}

void SGD::zero_grad() {
    for(auto &p : params)
        for(float &g : p->grad)
            g = 0.0f;
}


/* ===== operations ===== */

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {

    int rows = a->shape[0];
    int cols = a->shape[1];

    std::vector<float> out_data(rows*cols);

    // Check if b needs broadcasting (bias vector case)
    bool broadcast = (b->data.size() == cols);

    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++){
            float b_val = broadcast ? b->data[j] : b->data[i*cols+j];
            out_data[i*cols+j] = a->data[i*cols+j] + b_val;
        }

    auto out = std::make_shared<Tensor>(out_data, a->shape, a->requires_grad || b->requires_grad);

    if(out->requires_grad){
        out->parents = {a,b};

        Tensor* out_ptr = out.get();

        out->backward_fn = [a,b,out_ptr,rows,cols,broadcast](){

            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    float g = out_ptr->grad[i*cols+j];
                    a->grad[i*cols+j] += g;
                    
                    if(broadcast)
                        b->grad[j] += g;  // accumulate across batch
                    else
                        b->grad[i*cols+j] += g;
                }
            }
        };
    }

    return out;
}

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::vector<float> out_data(a->data.size());

    for (size_t i=0;i<a->data.size();i++)
        out_data[i] = a->data[i] * b->data[i];

    auto out = std::make_shared<Tensor>(out_data, a->shape, a->requires_grad || b->requires_grad);

    if (out->requires_grad) {
        out->parents = {a,b};

        Tensor* out_ptr = out.get();

        out->backward_fn = [a,b,out_ptr]() {
            for (size_t i=0;i<a->data.size();i++) {
                a->grad[i] += out_ptr->grad[i] * b->data[i];
                b->grad[i] += out_ptr->grad[i] * a->data[i];
            }

        };
    }

    return out;
}

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a) {
    float s = 0;
    for (float v : a->data) s += v;

   auto out = std::make_shared<Tensor>(std::vector<float>{s}, std::vector<int>{1}, a->requires_grad);

    if (out->requires_grad) {
        out->parents = {a};

        Tensor* out_ptr = out.get();

        out->backward_fn = [a,out_ptr]() {
            for (size_t i=0;i<a->data.size();i++)
                a->grad[i] += out_ptr->grad[0];
           // a->backward();
        };
    }

    return out;
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B) {

    int m = A->shape[0];
    int n = A->shape[1];
    int p = B->shape[1];

    std::vector<float> out_data(m*p, 0.0f);

    // Forward: C = A @ B
    //Paralleize
    #pragma omp parallel for collapse(2)
    for(int i=0;i<m;i++) {
        for(int j=0;j<p;j++) {
            float sum=0.0f;
            for(int k=0;k<n;k++)
                sum += A->data[i*n + k] * B->data[k*p + j];
            out_data[i*p + j] = sum;
        } }

    auto out = std::make_shared<Tensor>(out_data, std::vector<int>{m,p}, A->requires_grad || B->requires_grad);

    if(out->requires_grad){
        out->parents = {A,B};

        Tensor* out_ptr = out.get();

        out->backward_fn = [A,B,out_ptr,m,n,p](){

            // dA = dC @ Bᵀ
            for(int i=0;i<m;i++)
                for(int k=0;k<n;k++)
                    for(int j=0;j<p;j++)
                        A->grad[i*n + k] += out_ptr->grad[i*p + j] * B->data[k*p + j];

            // dB = Aᵀ @ dC
            for(int k=0;k<n;k++)
                for(int j=0;j<p;j++)
                    for(int i=0;i<m;i++)
                        B->grad[k*p + j] += A->data[i*n + k] * out_ptr->grad[i*p + j];

            //A->backward();
            //B->backward();
        };
    }

    return out;
}



/* ===== Linear Layer ===== */

Linear::Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f) {

    std::vector<float> w_data(in_f*out_f);
    std::vector<float> b_data(out_f, 0.1f);

    // simple random initialization
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f,0.1f);

    for(float &v : w_data) v = dist(gen);

    W = std::make_shared<Tensor>(w_data, std::vector<int>{in_f,out_f}, true);
    b = std::make_shared<Tensor>(b_data, std::vector<int>{1,out_f}, true);
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> x){

    auto y = matmul(x, W);     // xW
    y = add(y, b);             // + bias (broadcast later improved)

    return y;
}


/* ===== ReLU Activation ===== */

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> x) {

    std::vector<float> out_data(x->data.size());
    std::vector<float> mask(x->data.size());

    for(size_t i=0;i<x->data.size();i++){
        if(x->data[i] > 0){
            out_data[i] = x->data[i];
            mask[i] = 1.0f;
        } else {
            out_data[i] = 0.0f;
            mask[i] = 0.0f;
        }
    }

    auto out = std::make_shared<Tensor>(out_data, x->shape, x->requires_grad);

    if(out->requires_grad){
        out->parents = {x};

        Tensor* out_ptr = out.get();
        
        out->backward_fn = [x,out_ptr,mask](){

            for(size_t i=0;i<x->data.size();i++)
                x->grad[i] += out_ptr->grad[i] * mask[i];

            //x->backward();
        };
    }

    return out;
}


/* ===== Cross Entropy Loss ===== */
std::shared_ptr<Tensor> cross_entropy(
    std::shared_ptr<Tensor> logits,
    std::vector<int> targets)
{
    int N = logits->shape[0];
    int K = logits->shape[1];

    float loss_val = 0.0f;

    std::vector<float> max_vals(N);
    std::vector<float> sum_exps(N);

    // ===== forward =====
    for(int i=0;i<N;i++){
        float maxv = logits->data[i*K];
        for(int j=1;j<K;j++)
            maxv = std::max(maxv, logits->data[i*K+j]);
        max_vals[i]=maxv;

        float s=0;
        for(int j=0;j<K;j++)
            s+=std::exp(logits->data[i*K+j]-maxv);
        sum_exps[i]=s;

        loss_val += -logits->data[i*K+targets[i]] + std::log(s) + maxv;
    }

    loss_val /= N;

    auto out = std::make_shared<Tensor>(
        std::vector<float>{loss_val},
        std::vector<int>{1},
        logits->requires_grad
    );

    // ===== backward =====
    if(out->requires_grad){
        out->parents={logits};

        Tensor* out_ptr = out.get();

        out->backward_fn=[logits,out_ptr,targets,N,K,max_vals,sum_exps](){

            for(int i=0;i<N;i++){
                for(int j=0;j<K;j++){

                    float softmax =
                        std::exp(logits->data[i*K+j]-max_vals[i]) / sum_exps[i];

                    float grad = (softmax - (j==targets[i]?1.0f:0.0f))/N;

                    logits->grad[i*K+j] += grad * out_ptr->grad[0];
                }
            }
        };
    }

    return out;
}


inline int idx4d(int n,int c,int h,int w,
                 int C,int H,int W)
{
    return n*C*H*W + c*H*W + h*W + w;
}

inline int kidx4d(int f,int c,int kh,int kw,
                  int Cout,int Cin,int Kh,int Kw)
{
    return f*Cin*Kh*Kw + c*Kh*Kw + kh*Kw + kw;
}


Conv2D::Conv2D(int in_c, int out_c, int kh, int kw, int pad,int s)
    : in_channels(in_c), out_channels(out_c), kernel_h(kh), kernel_w(kw), padding(pad), stride(s)
{
    std::vector<float> w_data(out_c * in_c * kh * kw);
    std::vector<float> b_data(out_c, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f,0.1f);

    for(float &v : w_data) v = dist(gen);

    W = std::make_shared<Tensor>(
        w_data,
        std::vector<int>{out_c,in_c,kh,kw},
        true
    );

    b = std::make_shared<Tensor>(
        b_data,
        std::vector<int>{out_c},
        true
    );
}

std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> x)
{
    
    if(padding > 0)
        x = pad2d(x, padding);
    
    

    int N = x->shape[0];
    int C = x->shape[1];
    int H = x->shape[2];
    int W_in = x->shape[3];

    int H_out = (H - kernel_h)/stride + 1;
    int W_out = (W_in - kernel_w)/stride + 1;

    std::vector<float> out_data(N * out_channels * H_out * W_out, 0.0f);

    // ===== FORWARD =====
    //Paralleize
    #pragma omp parallel for collapse(2)
    for(int n=0;n<N;n++)
    for(int f=0;f<out_channels;f++)
    for(int i=0;i<H_out;i++)
    for(int j=0;j<W_out;j++)
    {
        float sum = b->data[f];

        for(int c=0;c<in_channels;c++)
        for(int kh=0;kh<kernel_h;kh++)
        for(int kw=0;kw<kernel_w;kw++)
        {
            int xi = i*stride + kh;
            int xj = j*stride + kw;

            int x_idx = idx4d(n,c,xi,xj,C,H,W_in);
            int w_idx = kidx4d(f,c,kh,kw,out_channels,in_channels,kernel_h,kernel_w)
;

            sum += x->data[x_idx] * W->data[w_idx];
        }

        int out_idx = ((n*out_channels + f)*H_out + i)*W_out + j;
        out_data[out_idx] = sum;
    }

    auto out = std::make_shared<Tensor>(
        out_data,
        std::vector<int>{N,out_channels,H_out,W_out},
        x->requires_grad || W->requires_grad || b->requires_grad
    );

    // ===== BACKWARD =====
    if(out->requires_grad){
        out->parents = {x,W,b};

        Tensor* out_ptr = out.get();
        
        out->backward_fn = [x, this, out_ptr, N, C, H, W_in, H_out, W_out]() {

            // ---- bias grad ----
            for(int n=0;n<N;n++)
            for(int f=0;f<out_channels;f++)
            for(int i=0;i<H_out;i++)
            for(int j=0;j<W_out;j++){
                int out_idx = ((n*out_channels+f)*H_out+i)*W_out+j;
                this->b->grad[f] += out_ptr->grad[out_idx];
            }

            // ---- weight grad ----
            for(int f=0;f<out_channels;f++)
            for(int c=0;c<in_channels;c++)
            for(int kh=0;kh<kernel_h;kh++)
            for(int kw=0;kw<kernel_w;kw++){

                float sum=0;

                for(int n=0;n<N;n++)
                for(int i=0;i<H_out;i++)
                for(int j=0;j<W_out;j++){

                    int xi=i*stride + kh;
                    int xj=j*stride + kw;

                    int x_idx=idx4d(n,c,xi,xj,C,H,W_in);
                    int out_idx=((n*out_channels+f)*H_out+i)*W_out+j;

                    sum += x->data[x_idx]*out_ptr->grad[out_idx];
                }

                int w_idx=kidx4d(f,c,kh,kw,out_channels,in_channels,kernel_h,kernel_w);
                this->W->grad[w_idx]+=sum;
            }

            // ---- input grad ----
            for(int n=0;n<N;n++)
            for(int c=0;c<in_channels;c++)
            for(int i=0;i<H;i++)
            for(int j=0;j<W_in;j++){

                float sum=0;

                for(int f=0;f<out_channels;f++)
                for(int kh=0;kh<kernel_h;kh++)
                for(int kw=0;kw<kernel_w;kw++){

                    int oi=i-kh;
                    int oj=j-kw;

                    if(oi%stride==0 && oj%stride==0){
                        oi/=stride;
                        oj/=stride;
                    }

                    if(oi>=0 && oj>=0 && oi<H_out && oj<W_out){
                        int out_idx=((n*out_channels+f)*H_out+oi)*W_out+oj;
                        int w_idx=kidx4d(f,c,kh,kw,out_channels,in_channels,kernel_h,kernel_w);
                        sum += this->W->data[w_idx]*out_ptr->grad[out_idx];
                    }
                }

                int x_idx=idx4d(n,c,i,j,C,H,W_in);
                x->grad[x_idx]+=sum;
            }
        };
    }

    return out;
}



MaxPool2D::MaxPool2D(int k) : kernel(k) {}

std::shared_ptr<Tensor> MaxPool2D::forward(std::shared_ptr<Tensor> x)
{
    int N = x->shape[0];
    int C = x->shape[1];
    int H = x->shape[2];
    int W = x->shape[3];

    int H_out = H / kernel;
    int W_out = W / kernel;

    std::vector<float> out_data(N*C*H_out*W_out,0.0f);

    // store max locations for backward
    std::vector<int> max_index(N*C*H_out*W_out,0);
    //Paralleize
    #pragma omp parallel for collapse(2)
    for(int n=0;n<N;n++)
    for(int c=0;c<C;c++)
    for(int i=0;i<H_out;i++)
    for(int j=0;j<W_out;j++)
    {
        float best = -1e9;
        int best_idx = 0;

        for(int kh=0;kh<kernel;kh++)
        for(int kw=0;kw<kernel;kw++)
        {
            int xi = i*kernel + kh;
            int xj = j*kernel + kw;

            int idx = idx4d(n,c,xi,xj,C,H,W);

            if(x->data[idx] > best){
                best = x->data[idx];
                best_idx = idx;
            }
        }

        int out_idx = ((n*C+c)*H_out+i)*W_out+j;
        out_data[out_idx] = best;
        max_index[out_idx] = best_idx;
    }

    auto out = std::make_shared<Tensor>(
    out_data,
    std::vector<int>{N,C,H_out,W_out},
    x->requires_grad
);

if(out->requires_grad){
    out->parents = {x};
    Tensor* out_ptr = out.get();
    out->backward_fn = [x,out_ptr,max_index,N,C,H,W,H_out,W_out,this]() {

        for(int i=0;i<x->grad.size();i++)
            x->grad[i]=0.0f;

        for(int n=0;n<N;n++)
        for(int c=0;c<C;c++)
        for(int i=0;i<H_out;i++)
        for(int j=0;j<W_out;j++)
        {
            int out_idx = ((n*C+c)*H_out+i)*W_out+j;
            int in_idx = max_index[out_idx];

            x->grad[in_idx] += out_ptr->grad[out_idx];
        }
    };
}

return out;


    
   
}


std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> x)
{
    int N = x->shape[0];
    int C = x->shape[1];
    int H = x->shape[2];
    int W = x->shape[3];

    int F = C*H*W;

    auto out = std::make_shared<Tensor>(
        x->data,
        std::vector<int>{N,F},
        x->requires_grad
    );

    if(out->requires_grad){
        out->parents = {x};

        Tensor* out_ptr = out.get();

        out->backward_fn = [x,out_ptr,N,C,H,W]() {

            for(size_t i=0;i<x->grad.size();i++)
                x->grad[i]+=out_ptr->grad[i];
        };
    }

    return out;
}

std::shared_ptr<Tensor> pad2d(std::shared_ptr<Tensor> x, int pad)
{
    if(pad==0) return x;

    int N=x->shape[0];
    int C=x->shape[1];
    int H=x->shape[2];
    int W=x->shape[3];

    int H2=H+2*pad;
    int W2=W+2*pad;

    std::vector<float> out_data(N*C*H2*W2,0.0f);

    for(int n=0;n<N;n++)
    for(int c=0;c<C;c++)
    for(int i=0;i<H;i++)
    for(int j=0;j<W;j++){
        int src=idx4d(n,c,i,j,C,H,W);
        int dst=idx4d(n,c,i+pad,j+pad,C,H2,W2);
        out_data[dst]=x->data[src];
    }

    auto out=std::make_shared<Tensor>(
        out_data,
        std::vector<int>{N,C,H2,W2},
        x->requires_grad
    );

    if(out->requires_grad){
        out->parents={x};

        Tensor* out_ptr = out.get();

        out->backward_fn=[x,out_ptr,N,C,H,W,H2,W2,pad](){

            for(int n=0;n<N;n++)
            for(int c=0;c<C;c++)
            for(int i=0;i<H;i++)
            for(int j=0;j<W;j++){
                int src=idx4d(n,c,i+pad,j+pad,C,H2,W2);
                int dst=idx4d(n,c,i,j,C,H,W);
                x->grad[dst]+=out_ptr->grad[src];
            }
        };
    }

    return out;
}
