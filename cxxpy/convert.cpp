#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <pybind11/complex.h>
//#include <pybind11/functional.h>
//#include <pybind11/chrono.h>
#include <openbabel/mol.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <variant>

namespace py = pybind11;

class CXXConverter {
public:
    // Function to convert Python list to C++ vector
    template <typename T>
    std::vector<T> list_to_cxxvector(const py::list& lst) {
        std::vector<T> result;
        for (size_t i = 0; i < lst.size(); ++i) {
            result.push_back(lst[i].cast<T>());
        }
        return result;
    }

    // Overloaded function to manage pointer types
    template <typename T>
    std::vector<T*> list_to_cxxvector_pointer(const py::list& lst) {
        std::vector<T*> result;
        for (size_t i = 0; i < lst.size(); ++i) {
            T* ptr = new T(lst[i].cast<T>());
            result.push_back(ptr);
        }
        return result;
    }

    // Function to wrap the type conversion
    std::vector<double> convert_to_double_vector(const py::list& lst) {
        return list_to_cxxvector<double>(lst);
    }

    std::vector<int> convert_to_int_vector(const py::list& lst) {
        return list_to_cxxvector<int>(lst);
    }

    std::vector<double*> convert_to_double_ptr_vector(const py::list& lst) {
        return list_to_cxxvector_pointer<double>(lst);
    }

    std::vector<int*> convert_to_int_ptr_vector(const py::list& lst) {
        return list_to_cxxvector_pointer<int>(lst);
    }

    // Destructor to clean up pointer vectors
    ~CXXConverter() {
        // Clean up any allocated memory if needed
        for (auto ptr : double_pointers) {
            delete ptr;
        }
        for (auto ptr : int_pointers) {
            delete ptr;
        }
    }

    void set_double_pointers(const std::vector<double*>& pointers) {
        double_pointers = pointers;
    }

    void set_int_pointers(const std::vector<int*>& pointers) {
        int_pointers = pointers;
    }

private:
    std::vector<double*> double_pointers;
    std::vector<int*> int_pointers;
};

// Binding using pybind11
using variant_vector = std::variant<std::vector<int>, std::vector<double>>;
PYBIND11_MODULE(cxxconvert, m) {
    py::class_<CXXConverter>(m, "CXXConverter")
        .def(py::init<>())
        .def("list_to_cxxvector_double", &CXXConverter::list_to_cxxvector<double>)
        .def("list_to_cxxvector_int", &CXXConverter::list_to_cxxvector<int>)
        .def("list_to_cxxvector_double_pointer", &CXXConverter::list_to_cxxvector_pointer<double>)
        .def("list_to_cxxvector_int_pointer", &CXXConverter::list_to_cxxvector_pointer<int>)
        .def("list_to_cxxvector", [](CXXConverter& converter, const py::list& lst, const std::string& dtype) -> variant_vector {
            if (dtype == "double") {
                return converter.list_to_cxxvector<double>(lst);
            } else if (dtype == "int") {
                return converter.list_to_cxxvector<int>(lst);
            } else {
                throw std::invalid_argument("Unsupported dtype: " + dtype);
            }
        });
}

//PYBIND11_MODULE(cxxconvert, m) {
//    py::class_<CXXConverter>(m, "CXXConverter")
//        .def(py::init<>())
//        .def("list_to_cxxvector", [](CXXConverter& converter, const py::list& lst, const std::string& dtype) {
//            if (dtype == "double") {
//                return converter.list_to_cxxvector<double>(lst);
//            } else if (dtype == "int") {
//                return converter.list_to_cxxvector<int>(lst);
//            } else if (dtype == "double*") {
//                return converter.list_to_cxxvector_pointer<double>(lst);
//            } else if (dtype == "int*") {
//                return converter.list_to_cxxvector_pointer<int>(lst);
//            } else {
//                throw std::invalid_argument("Unsupported dtype: " + dtype);
//            }
//        });
//}