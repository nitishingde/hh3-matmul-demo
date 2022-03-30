// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of 
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.


#ifndef TUTORIAL3_OUTPUT_STATE_H
#define TUTORIAL3_OUTPUT_STATE_H
#include <hedgehog/hedgehog.h>
#include <ostream>
#include "../data/data_type.h"
#include "../data/matrix_block_data.h"

template<class Type, Order Ord = Order::Row>
class OutputState : public hh::State<1, MatrixBlockData<Type, 'c', Ord>, MatrixBlockData<Type, 'c', Ord>> {
 private:
  std::vector<size_t>
      ttl_ = {};

  size_t
      gridHeightTTL_ = 0,
      gridWidthTTL_ = 0;

 public:
  OutputState(size_t gridHeightTtl, size_t gridWidthTtl, size_t const &ttl)
      : ttl_(std::vector<size_t>(gridHeightTtl * gridWidthTtl, ttl)), 
      gridHeightTTL_(gridHeightTtl), gridWidthTTL_(gridWidthTtl) {}

  virtual ~OutputState() = default;

  void execute(std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> ptr) override {
    auto i = ptr->rowIdx(), j = ptr->colIdx();
    --ttl_[i * gridWidthTTL_ + j];
    if(ttl_[i * gridWidthTTL_ + j] == 0){
      this->push(ptr);
    }
  }

  friend std::ostream &operator<<(std::ostream &os, OutputState const &state) {
    for(size_t i = 0; i < state.gridHeightTTL_; ++i){
      for(size_t j = 0; j < state.gridWidthTTL_; ++j) {
        os << state.ttl_[i * state.gridWidthTTL_ + j] << " ";
      }
      os << std::endl;
    }
    return os;
  }
};

#endif //TUTORIAL3_OUTPUT_STATE_H
