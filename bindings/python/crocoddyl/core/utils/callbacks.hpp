///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
#define PYTHON_CROCODDYL_CORE_UTILS_CALLBACKS_HPP_

#include "crocoddyl/core/utils/callbacks.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCallbacks() {
  bp::class_<CallbackVerbose, bp::bases<CallbackAbstract> >("CallbackVerbose",
                                                            "Callback function for printing the solver values.")
      .def("__call__", &CallbackVerbose::operator(), bp::args(" self", " solver"),
           "Run the callback function given a solver.\n\n"
           ":param solver: solver to be diagnostic");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_UTILS_CALLBACKS_HPP_