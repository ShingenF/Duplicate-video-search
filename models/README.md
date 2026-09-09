# Local AI Models

Put ONNX vision embedding models in this folder and reference them from Settings, for example:

```text
models\dinov2\model.onnx
```

Model weight files are intentionally ignored by git because they can be large. The V2 AI index stores only local frame embeddings in `data\index.sqlite`; video frames and embeddings are not uploaded.

Recommended starting point: a DINOv2 or CLIP-style image embedding model exported to ONNX with input shape `N x 3 x 224 x 224`.

The default model path is:

```text
models\dinov2-small-dynamic\model.onnx
```

It is downloaded from `Xenova/dinov2-small` on Hugging Face (`onnx/model.onnx`) and has a dynamic batch input named `pixel_values`, so the app can run AI indexing with batch sizes such as 16 or 32.

Download the unquantized [model.onnx](https://huggingface.co/Xenova/dinov2-small/blob/main/onnx/model.onnx) yourself; the application does not automatically download it. Review the model publisher's license before use or redistribution. Do not rename an arbitrary model or quantized variant and assume compatibility.

The current pipeline supplies float32 RGB NCHW images, resized to 224×224 and normalized with ImageNet mean/std. It reads the first output tensor as embeddings. Different preprocessing, input types, multiple required inputs, or incompatible output shapes need code changes; not every ONNX or CLIP model is supported. Start with the default DINOv2 model. No model inference was required for the unit-test suite.
