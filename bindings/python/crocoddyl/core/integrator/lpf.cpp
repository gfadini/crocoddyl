///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/integrator/lpf.hpp"

namespace crocoddyl {
namespace python {

void exposeIntegratedActionLPF() {
  bp::class_<IntegratedActionModelLPF, bp::bases<ActionModelAbstract> >(
      "IntegratedActionModelLPF",
      "Sympletic LPF integrator for differential action models.\n\n"
      "This class implements a sympletic LPF integrator (a.k.a semi-implicit\n"
      "integrator) give a differential action model, i.e.:\n"
      "  [q+, v+] = State.integrate([q, v], [v + a * dt, a * dt] * dt).",
      bp::init<boost::shared_ptr<DifferentialActionModelAbstract>, bp::optional<int, double, bool, double> >(
          bp::args("self", "diffModel", "nu", "stepTime", "withCostResidual", "alpha"),
          "Initialize the sympletic LPF integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param stepTime: step time\n"
          ":param withCostResidual: includes the cost residuals and derivatives\n"
          ":param alpha: forgetting factor of the filter."))
      .def<void (IntegratedActionModelLPF::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &IntegratedActionModelLPF::calc, bp::args("self", "data", "x", "u"),
          "Compute the time-discrete evolution of a differential action model.\n\n"
          "It describes the time-discrete evolution of action model.\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input")
      .def<void (IntegratedActionModelLPF::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (IntegratedActionModelLPF::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &IntegratedActionModelLPF::calcDiff, bp::args("self", "data", "x", "u"),
          "Computes the derivatives of the integrated action model wrt state and control. \n\n"
          "It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input\n")
      .def<void (IntegratedActionModelLPF::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &IntegratedActionModelLPF::createData, bp::args("self"),
           "Create the LPF integrator data.")
      .def("setAlpha", &IntegratedActionModelLPF::setAlpha, bp::args("f_c"),
           "Sets alpha from the cut off frequency f_c.")
      .add_property("differential",
                    bp::make_function(&IntegratedActionModelLPF::get_differential,
                                      bp::return_value_policy<bp::return_by_value>()),
                    &IntegratedActionModelLPF::set_differential, "differential action model")
      .add_property("alpha",
                    bp::make_function(&IntegratedActionModelLPF::get_alpha,
                                      bp::return_value_policy<bp::return_by_value>()),
                    &IntegratedActionModelLPF::set_alpha, "alpha")
      .add_property(
          "dt", bp::make_function(&IntegratedActionModelLPF::get_dt, bp::return_value_policy<bp::return_by_value>()),
          &IntegratedActionModelLPF::set_dt, "step time");

  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionDataLPF> >();

  bp::class_<IntegratedActionDataLPF, bp::bases<ActionDataAbstract> >(
      "IntegratedActionDataLPF", "Sympletic LPF integrator data.",
      bp::init<IntegratedActionModelLPF*>(bp::args("self", "model"),
                                            "Create sympletic LPF integrator data.\n\n"
                                            ":param model: sympletic LPF integrator model"))
      .add_property(
          "differential",
          bp::make_getter(&IntegratedActionDataLPF::differential, bp::return_value_policy<bp::return_by_value>()),
          "differential action data")
      .add_property("dx", bp::make_getter(&IntegratedActionDataLPF::dx, bp::return_internal_reference<>()),
                    "state rate.");
}

}  // namespace python
}  // namespace crocoddyl
