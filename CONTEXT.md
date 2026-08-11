# StructCast-Model

Constructs neural network models and training workflows from declarative configuration. This glossary defines the
terms that are specific to this project; general Python and machine-learning vocabulary is not repeated here.

## Language

**INPUT_SHAPES**:
A user-defined layer's declaration of what its named inputs look like. It is carried into the generated model so the
model can describe its own inputs, instead of the caller having to state them again on the command line.
_Avoid_: input spec, shape config, dummy input config

**TensorSpec**:
The description of a single tensor input — its shape, its element type, and optionally the initializer that fills it.
It is always a description, never a tensor.
_Avoid_: tensor config, shape entry, input descriptor

**TensorSpecTree**:
A `TensorSpec`, or a nested dictionary or list of them, mirroring the structure in which a layer expects its inputs.
_Avoid_: nested shapes, shape tree, input structure

**TensorInitializer**:
A callable that produces a concrete tensor of a requested size and element type. Either the one a `TensorSpec` names,
or the framework's default when the spec names none.
_Avoid_: filler, generator, factory
