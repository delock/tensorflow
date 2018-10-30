/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class SamplingDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit SamplingDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    // Create a new SamplingDatasetOp::Dataset, and return it as the output.
    float rate;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<float>(ctx, "rate", &rate));
    *output = new Dataset(ctx, rate, input);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, float rate, const DatasetBase* input)
        : GraphDatasetBase(ctx), rate_(rate), input_(input) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      if (rate_ == 0) {
        assert (!"rate cannot be 0");
      } else {
        return std::unique_ptr<IteratorBase>(new Iterator(
            {this, strings::StrCat(prefix, "::Sampling")}));
      }
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "SamplingDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph_node));
      Node* rate = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(rate_, &rate));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {input_graph_node, rate}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // NOTE(mrry): This method is thread-safe as long as
        // `input_impl_` and `f` are thread-safe. However, if multiple
        // threads enter this method, outputs may be observed in a
        // non-deterministic order.
        bool matched;
        do {
          {
            tf_shared_lock l(mu_);
            if (!input_impl_) {
              *end_of_sequence = true;
              return Status::OK();
            }
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
          }
          if (*end_of_sequence) {
            mutex_lock l(mu_);
            input_impl_.reset();
            return Status::OK();
          }

          // generating a number from random uniform [0, 1)
          int rand_val;
          while ((rand_val=std::rand())==RAND_MAX)
            ;
          matched = (float)(rand_val)/(float)(RAND_MAX) < dataset()->rate_;
          if (!matched) {
            // Clear the output tensor list since it didn't match.
            out_tensors->clear();
          }
        } while (!matched);
        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (input_impl_) {
          TF_RETURN_IF_ERROR(SaveParent(writer, input_impl_));
        } else {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (!reader->Contains(full_name("input_impl_empty"))) {
          TF_RETURN_IF_ERROR(RestoreParent(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const float rate_;
    const DatasetBase* const input_;
  };
};

REGISTER_KERNEL_BUILDER(Name("SamplingDataset").Device(DEVICE_CPU), SamplingDatasetOp);

}  // namespace

}  // namespace tensorflow
