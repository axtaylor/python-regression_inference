#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>
#include "regression.h"

namespace py = pybind11;

class LinearRegression {

private:
    bool fitted; 
    std::vector<float> theta;   // Vector of [Bias, Weight... Weight]              
    int n;                      // shape[0]
    int k;                      // shape[1]
    
public:
    LinearRegression() : fitted(false), n(0), k(0) {}
    
    // Mandatory arguments to be received from Python API
    void fit(
        py::array_t<float> X,   // np.ndarray()
        py::array_t<float> y,   // np.ndarray()
        int max_iterations,
        float lr,
        float tol
    )
    {
        py::buffer_info X_buffer = X.request(); // Call array commands on pybuffer object
        py::buffer_info y_buffer = y.request();
        
        n = X_buffer.shape[0];
        k = X_buffer.shape[1];

        theta.resize(k); // Allocate theta vector shape of feature row
        
        // __fit.cu
        __fit(
            static_cast<float*>(X_buffer.ptr),  // *(float)X_buffer.ptr
            static_cast<float*>(y_buffer.ptr),
            max_iterations,
            lr,
            tol,
            theta.data(),                       // Call by reference
            n,
            k
        );
        fitted = true;
    }

    py::array_t<float> predict(py::array_t<float> X) {

        if (!fitted) {
            throw std::runtime_error("Model is not fitted. Call fit() with arguments before calling predict().");
        }

        py::buffer_info X_buffer = X.request();

        int n_prediction = X_buffer.shape[0];

        if (X_buffer.ndim != 2) {
            throw std::runtime_error("Expected 2D array: np.array( [[]] ).");
        }
        if (X_buffer.shape[1] != k) {
            throw std::runtime_error("Feature Count != Model Feature Count. Include a constant term.");
        }

        py::array_t<float> result = py::array_t<float>(n_prediction); // Prediction space of n length * k features
        py::buffer_info prediction = result.request();

        // __predict.cu
        __predict(
            static_cast<float*>(X_buffer.ptr),    
            theta.data(),                             
            static_cast<float*>(prediction.ptr),
            n_prediction,  
            k
        );

        return result;
    }
    
    bool isFitted() const {
        return fitted;
    }

    std::vector<float> getTheta() const {
        return theta;
    }  
};


PYBIND11_MODULE(CRegression, handle)    // Allow python access to LinearRegression()
{
    handle.doc() = "CUDA accelerated Gradient Descent Linear Regression";
    
    py::class_<LinearRegression>(handle, "LinearRegression")

        .def(py::init<>())  // Constructor is required to call the class

        .def("fit", &LinearRegression::fit,
            py::arg("X"),
            py::arg("y"),
            py::arg("max_iterations"),
            py::arg("lr"),
            py::arg("tol")
        ) 

        .def(
            "predict",
            &LinearRegression::predict,
            py::arg("X")
        )

        .def("is_fitted", &LinearRegression::isFitted)  
        .def_property_readonly("theta", &LinearRegression::getTheta);
}