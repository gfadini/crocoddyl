///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "crocoddyl_state_factory.hpp"

#ifndef CROCODDYL_IMPULSES_FACTORY_HPP_
#define CROCODDYL_IMPULSES_FACTORY_HPP_

namespace crocoddyl_unit_test {

struct ImpulseModelTypes {
  enum Type { ImpulseModel3DTalosArm,
              ImpulseModel3DHyQ,
              ImpulseModel3DTalos,
              ImpulseModel3DRandomHumanoid,
              ImpulseModel6DTalosArm,
              ImpulseModel6DHyQ,
              ImpulseModel6DTalos,
              ImpulseModel6DRandomHumanoid,
              NbImpulseModelTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbImpulseModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<ImpulseModelTypes::Type> ImpulseModelTypes::all(ImpulseModelTypes::init_all());

class ImpulseModelFactory {
 public:
  ImpulseModelFactory(ImpulseModelTypes::Type type) {
    test_type_ = type;
    
    size_t frame = 0;
    boost::shared_ptr<crocoddyl::StateMultibody> state;

    switch (test_type_) {
      case ImpulseModelTypes::ImpulseModel3DTalosArm:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyTalosArm);
        std::cout << "gripper_left_base_link" << std::endl;
        frame = state_factory_->get_pinocchio_model().getFrameId("gripper_left_base_link");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        model_.reset(new crocoddyl::ImpulseModel3D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel3DHyQ:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyHyQ);
        std::cout << "lf_foot" << std::endl;
        frame = state_factory_->get_pinocchio_model().getFrameId("lf_foot");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        model_.reset(new crocoddyl::ImpulseModel3D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel3DTalos:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyTalos);
        std::cout << "gripper_left_base_link" << std::endl;
        frame = state_factory_->get_pinocchio_model().getFrameId("gripper_left_base_link");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        model_.reset(new crocoddyl::ImpulseModel3D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel3DRandomHumanoid:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyRandomHumanoid);
        std::cout << "rleg6" << std::endl;
        frame = state_factory_->get_pinocchio_model().getFrameId("rleg6_body");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        model_.reset(new crocoddyl::ImpulseModel3D(state, frame));
        break;

      case ImpulseModelTypes::ImpulseModel6DTalosArm:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyTalosArm);
        std::cout << "gripper_left_base_link" << std::endl;
        frame = state_factory_->get_pinocchio_model().getFrameId("gripper_left_base_link");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        model_.reset(new crocoddyl::ImpulseModel6D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel6DHyQ:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyHyQ);
        std::cout << "lf_foot" << std::endl;
        frame = state_factory_->get_pinocchio_model().getFrameId("lf_foot");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        model_.reset(new crocoddyl::ImpulseModel6D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel6DTalos:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyTalos);
        std::cout << "gripper_left_base_link" << std::endl;
        frame = state_factory_->get_pinocchio_model().getFrameId("gripper_left_base_link");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        model_.reset(new crocoddyl::ImpulseModel6D(state, frame));
        break;
      case ImpulseModelTypes::ImpulseModel6DRandomHumanoid:
        state_factory_ = boost::make_shared<StateFactory>(StateTypes::StateMultibodyRandomHumanoid);
        std::cout << "rleg6" << std::endl;
        frame = state_factory_->get_pinocchio_model().getFrameId("rleg6_body");
        state = boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_->get_state());
        model_.reset(new crocoddyl::ImpulseModel6D(state, frame));
        break;

      default:
        throw std::runtime_error(__FILE__ ": Wrong ImpulseModelTypes::Type given");
        break;
    }
  }

  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> get_model() { return model_; }
  boost::shared_ptr<StateFactory> get_state_factory() { return state_factory_; }
  double num_diff_modifier_;

 private:
  ImpulseModelTypes::Type test_type_; //!< The type of impulse to test
  boost::shared_ptr<crocoddyl::ImpulseModelAbstract> model_; //!< The pointer to the impulse model
  boost::shared_ptr<StateFactory> state_factory_; //!< The pointer to the multibody state factory
};


} // namespace crocoddyl_unit_test

#endif // CROCODDYL_IMPULSES_FACTORY_HPP_
