#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_apple(py::module &&m) {
  m.doc() = "Triton Apple GPU backend bindings";
}
