#include <Python.h>
#include <iostream>

void detectObjects() {
    // Initialize Python interpreter
    Py_Initialize();

    // Import necessary modules
    PyObject* pName = PyUnicode_DecodeFSDefault("trial.py"); // Replace "your_python_module" with the name of your Python module
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        // Get the function
        PyObject* pFunc = PyObject_GetAttrString(pModule, "detect_objects");

        if (pFunc && PyCallable_Check(pFunc)) {
            // Call the Python function with arguments
            PyObject* pArgs = PyTuple_New(3); // Provide appropriate arguments here
            PyTuple_SetItem(pArgs, 0, PyUnicode_FromString("image_dir")); // Replace "image_dir" with your image directory
            PyTuple_SetItem(pArgs, 1, PyUnicode_FromString("output_dir")); // Replace "output_dir" with your output directory
            PyTuple_SetItem(pArgs, 2, PyUnicode_FromString("weights_path")); // Replace "weights_path" with your weights path

            PyObject* pResult = PyObject_CallObject(pFunc, pArgs);

            // Check for errors
            if (pResult != NULL) {
                // Do something with the result if needed
            } else {
                PyErr_Print();
            }

            // Clean up
            Py_XDECREF(pArgs);
            Py_XDECREF(pResult);
            Py_XDECREF(pFunc);
        } else {
            if (PyErr_Occurred()) PyErr_Print();
            std::cerr << "Cannot find function\n";
        }

        // Clean up
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        std::cerr << "Failed to load Python module\n";
    }

    // Finalize Python interpreter
    Py_Finalize();
}

int main() {
    // Call the detection function
    detectObjects();
    return 0;
}
