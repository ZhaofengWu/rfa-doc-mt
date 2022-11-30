#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <stdio.h>
typedef torch::Tensor Tensor;


Tensor RandomProject(
        Tensor const& x,
        Tensor const& w
);


std::vector<Tensor> RandomProject(
        Tensor const& x,
        Tensor const& y,
        Tensor const& w
);


Tensor RFAForward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v);


std::vector<Tensor> RFABackward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v,
        Tensor const& grad_attn);


std::vector<Tensor> CalculateSZ(
        Tensor const& k, 
        Tensor const& v
);


Tensor CrossRFA(
        Tensor const& q,
        Tensor const& s,
        Tensor const& z
);


std::vector<Tensor> CausalRFA(
        Tensor const& q, 
        Tensor const& k, 
        Tensor const& v,
        Tensor & s,
        Tensor & z
);


Tensor random_project(
        Tensor const& x,
        Tensor const& w) {
    return RandomProject(x, w);
}


std::vector<Tensor> random_project_xy(
        Tensor const& x,
        Tensor const& y,
        Tensor const& w) {
    return RandomProject(x, y, w);
}


std::vector<Tensor> calculate_sz(
        Tensor const& k, 
        Tensor const& v) {
    return CalculateSZ(k, v);
}



std::vector<Tensor> causal_rfa(
        Tensor const& q, 
        Tensor const& k, 
        Tensor const& v,
        Tensor & s,
        Tensor & z) {
    return CausalRFA(q, k, v, s, z);
}


Tensor cross_rfa(
        Tensor const& q,
        Tensor const& s,
        Tensor const& z) {
    return CrossRFA(q, s, z);
}


Tensor forward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v) {
    return RFAForward(q, k, v);
}


std::vector<Tensor> backward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v,
        Tensor const& grad_attn) {
    return RFABackward(q, k, v, grad_attn);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RFA Forward");
    m.def("backward", &backward, "RFA Backward");
    m.def("causal_rfa", &causal_rfa, "causal RFA");
    m.def("calculate_sz", &calculate_sz, "calculate sz");
    m.def("cross_rfa", &cross_rfa, "cross RFA");
    m.def("random_project", &random_project, "random project");
    m.def("random_project_xy", &random_project_xy, "random project");
}
