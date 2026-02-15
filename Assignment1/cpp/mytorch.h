#pragma once
#include <functional>
#include <memory>
#include <vector>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<int> shape;        // NEW
    bool requires_grad;
    bool visited = false;        // for backward pass

    std::vector<std::shared_ptr<Tensor>> parents;
    std::function<void()> backward_fn;

    Tensor(std::vector<float> values, std::vector<int> shape, bool requires_grad=false);

    void backward();
};


class Linear {
public:
    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;

    int in_features;
    int out_features;

    Linear(int in_f, int out_f);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

class SGD {
public:
    std::vector<std::shared_ptr<Tensor>> params;
    float lr;

    SGD(std::vector<std::shared_ptr<Tensor>> parameters, float learning_rate);

    void step();
    void zero_grad();
};

class Conv2D {
public:
    int in_channels, out_channels;
    int kernel_h, kernel_w;
    int padding;
    int stride;

    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;

    Conv2D(int in_c, int out_c, int kh, int kw, int pad=0, int stride=1);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

class MaxPool2D {
public:
    int kernel;

    MaxPool2D(int k=2);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

class Flatten {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};



/* operations */
std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> B);   // NEW
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> x);
std::shared_ptr<Tensor> cross_entropy(std::shared_ptr<Tensor> logits, std::vector<int> targets);
std::shared_ptr<Tensor> pad2d(std::shared_ptr<Tensor> x, int pad);

