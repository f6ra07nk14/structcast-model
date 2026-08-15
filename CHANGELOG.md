## [4.0.0](https://github.com/f6ra07nk14/structcast-model/compare/v3.0.0...v4.0.0) (2026-08-15)


### ⚠ BREAKING CHANGES

* **base-trainer:** lifecycle events no longer pass **models keyword
arguments; every handler signature is on_*(info) and
on_best(info, best), with models read from info.models. on_best's
criterion parameter tightens from BestCriterion[Any] to
BestCriterion[ModelT].

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
* **torch:** --resume must match the active --logger; resuming from the
other service's reference (e.g. --logger wandb with runs:/...) now fails
with a ValueError. Download the artifact and resume from a local path
instead.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

### 📔 Docs

* **reference:** correct CFG and TYPE sublayer semantics ([fdb72be](https://github.com/f6ra07nk14/structcast-model/commit/fdb72bea43b67ce1e0d2f3d63a1b4aa1c9c06de5))


### 🔧 Fixes

* **base-trainer:** cache the provider step counts on first read ([25f4c2a](https://github.com/f6ra07nk14/structcast-model/commit/25f4c2a5e4b4a54460e9794d011f24df82853f9f))


### 🔨 Refactor

* **base-trainer:** move models onto the info object ([38e05dd](https://github.com/f6ra07nk14/structcast-model/commit/38e05dd543662380a1a17351f425c9a4978d0c0a))
* **builders:** move resolver utilities ([c42f5a1](https://github.com/f6ra07nk14/structcast-model/commit/c42f5a18bdfda4c7acfae2e2680fafff88d990e8))
* **torch:** move training-state fetching into the loggers ([a4daee6](https://github.com/f6ra07nk14/structcast-model/commit/a4daee6f58303d7111a26197ad5e838105147fbe))


### 🚨 Tests

* use canonical cfg configs and move fixtures under tests/fixtures ([c58a5fa](https://github.com/f6ra07nk14/structcast-model/commit/c58a5fa2bf01648780dda21cfe05e3fb2d747ccb))

## [3.0.0](https://github.com/f6ra07nk14/structcast-model/compare/v2.0.0...v3.0.0) (2026-08-15)


### ⚠ BREAKING CHANGES

* **torch:** TorchTrainer.no_sync, _unwrap_ddp, and _get_state_dict are
gone; TrainingStateSaver and TorchBestCriterion.from_criteria require a
strategy and accept a None logger; interrupt-time saving is removed.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
* **builders:** generated learner scripts changed shape (flow functions,
sync gates, __grad_scaler_creator__); MIXED_PRECISION with a non-float16
MIXED_PRECISION_TYPE is now a SpecError; bf16 learners expose empty
grad_scalers.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
* **torch:** TorchLearnerFactory is gone and the loggers import from new
module paths.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
* **base-trainer:** BaseTrainer requires data= at construction and
SimpleDataProvider takes keyword arguments only.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
* **torch:** the Timm* classes are no longer part of
structcast_model.torch.trainer, and train no longer logs timm_version.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
* **base-trainer:** objects implementing Learner must expose optimizers and
learning_rates properties.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
* **cli,builders:** the backward CLI vocabulary, BACKWARDS template key,
cfg/torch/backwards/ directory and create_with_scheduler cfg patterns are gone.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
* **torch:** create_with_scheduler and the scheduler registration side
channel are removed; TorchTracker.reset and the wrapper's set_dataset_epoch/
set_dataloader_epoch methods are replaced.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
* **base-trainer:** GLOBAL_CALLBACKS, callbacks_session, NamedCallbackList and
the Callbacks dataclass are removed; Backward is renamed Learner; fit() no
longer accepts datasets.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
* **builders:** unannotated dummy inputs are now bfloat16, not float32

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

### 💥 Breaking Changes

* **base-trainer:** require optimizers and learning_rates on the Learner protocol ([e48ce25](https://github.com/f6ra07nk14/structcast-model/commit/e48ce255498337857b1591d356ffbba153e300f8))
* **base-trainer:** require the data provider and keyword-only SimpleDataProvider ([ad6aebc](https://github.com/f6ra07nk14/structcast-model/commit/ad6aebc9288dc581e751e12aa5c04039d998296a))
* **base-trainer:** restore runtime-checkable protocols via typing_extensions ([d711c79](https://github.com/f6ra07nk14/structcast-model/commit/d711c79e4e64c75d501451bc328a235cb3a0eff8))
* **base-trainer:** route callbacks by protocol and rename Backward to Learner ([bd69016](https://github.com/f6ra07nk14/structcast-model/commit/bd6901680b835befa8689c15b8f7dad58d5e3c9c))
* **builders:** bind learner steps directly, dropping forward_* indirection ([58b4d7b](https://github.com/f6ra07nk14/structcast-model/commit/58b4d7bcd112a1da22a8862085250edb4db17aa5))
* **builders:** generate strategy-aware learners with flow functions and sync gates ([4429202](https://github.com/f6ra07nk14/structcast-model/commit/4429202d430a1f39a8310a05c0a077ff4358d5ad))
* **builders:** generated steps become class methods importing package helpers ([a20880e](https://github.com/f6ra07nk14/structcast-model/commit/a20880e6d2a721708ab4cfc95ed071abe7b01154))
* **cli,builders:** cascade Learner rename and rewire the train command ([c5f2b59](https://github.com/f6ra07nk14/structcast-model/commit/c5f2b59f25e098585498d7744265d1a607a3579a))
* **commands:** let the configuration decide structured output; compile in place ([cd50ead](https://github.com/f6ra07nk14/structcast-model/commit/cd50ead7f2f6b5858306a21d53c6a1964c858bcc))
* **example:** attention and RoPE live in the configuration; Transformer becomes SmallLanguageModel ([af84a9c](https://github.com/f6ra07nk14/structcast-model/commit/af84a9c7e38b30994e76121c740044f55e997320))
* **torch:** assemble training through the distributed strategy and add resume ([274f49c](https://github.com/f6ra07nk14/structcast-model/commit/274f49c102a51864dffea1f979c7dd2f340ed513))
* **torch:** move timm.data integrations to examples and make the CLI timm-agnostic ([6215eca](https://github.com/f6ra07nk14/structcast-model/commit/6215eca09adcbb11f07476d1831e6e310089bb0c))
* **torch:** move torch layer to protocol-routed Learner design ([fb7d44d](https://github.com/f6ra07nk14/structcast-model/commit/fb7d44d86bba85d8bd87da6046b580d6932511b4))
* **torch:** split loggers behind a Logger protocol and revert the learner factory ([89a402a](https://github.com/f6ra07nk14/structcast-model/commit/89a402a796c597d3d175164527a09eef08eb7c16))
* **trainer:** own step counts on the DataProvider and rework callback wiring ([25ddffc](https://github.com/f6ra07nk14/structcast-model/commit/25ddffcb0bdcec3b5be23e0f3943c7baca0b8b78))


### 👷 Build

* **docker:** add a GPU training image for DGX validation runs ([0df26cc](https://github.com/f6ra07nk14/structcast-model/commit/0df26cc0835c1c46fbea9003bb5409820809e178))


### 📦 Other

* add anysearch skill ([179186c](https://github.com/f6ra07nk14/structcast-model/commit/179186cd981782e32bc9da978a4639b1d3562b80))
* add devcontainer and agent skills ([e8e79fb](https://github.com/f6ra07nk14/structcast-model/commit/e8e79fbbbb3d997238fc76c2fa6f622baa4d4e89))
* refresh uv.lock for structcast 2.0 ([f8ffec3](https://github.com/f6ra07nk14/structcast-model/commit/f8ffec3d0742131ade5988dbb21dfadc22349807))
* remove anysearch skill ([bd6894c](https://github.com/f6ra07nk14/structcast-model/commit/bd6894ceed0e400570ac09a12686eef0551103ff))
* update .gitignore to exclude Claude Code settings and symlink ([7f9c490](https://github.com/f6ra07nk14/structcast-model/commit/7f9c4906ab0e44dc078f86ab976e1c8fe4aafa15))
* update devcontainer configuration by removing unused variables and scripts ([e0ca6cf](https://github.com/f6ra07nk14/structcast-model/commit/e0ca6cf2589b42c92d56f7e12af515b4d2c65efa))


### 📔 Docs

* add domain glossary and training-loop redesign ADR ([ab9ab52](https://github.com/f6ra07nk14/structcast-model/commit/ab9ab52c4292890016bde9bdbe39b179fc3b0681))
* Add StructCast-Model Reference documentation ([8c444cc](https://github.com/f6ra07nk14/structcast-model/commit/8c444ccd7470baf4b33f65755d3de000a9f4ad72))
* **adr:** record distributed strategy, sync gating, and checkpoint decisions ([6e739e5](https://github.com/f6ra07nk14/structcast-model/commit/6e739e5e35161b7e6c70e63eec643ed779c3e5e3))
* align distributed training claims with the strategy design ([d4a28e8](https://github.com/f6ra07nk14/structcast-model/commit/d4a28e83fe0aa44cfa90fbb7e2a57205faf44aa1))
* align guides with setter sync_gate, strategy compile, and NullLogger; drop redundant update_models ([2eca54e](https://github.com/f6ra07nk14/structcast-model/commit/2eca54ebea35e750885216661c8bad3164848881))
* align the docs with the current public API ([268ac76](https://github.com/f6ra07nk14/structcast-model/commit/268ac76208eb201755d2c99635983926f3bb19fc))
* **references:** bound the vision run-to-run spread and retract a compile regression ([7176ae9](https://github.com/f6ra07nk14/structcast-model/commit/7176ae9b6496518c1138c322f1207aae16deaf3d))
* **references:** complete the vision strategy matrix ([6919f0a](https://github.com/f6ra07nk14/structcast-model/commit/6919f0a33d2a0bb0b438794910defec7ced995c0))
* **references:** explain the ConvNeXt V2-B per-epoch cost and the compile relaunch ([4ebfd79](https://github.com/f6ra07nk14/structcast-model/commit/4ebfd79fa742d11b131032c0d5658044364df6ac))
* **references:** note the vision matrix runs a single seed ([4b3deee](https://github.com/f6ra07nk14/structcast-model/commit/4b3deeed07bbebc730ebf43749e86a81d66aa4a0))
* **references:** record per-block FSDP2 validation on H200 ([27c75b5](https://github.com/f6ra07nk14/structcast-model/commit/27c75b545c045d0d113b5af000d58c4b1a288cbf))
* **references:** record strategy comparison runs on H200 ([b2520cc](https://github.com/f6ra07nk14/structcast-model/commit/b2520cc39c3ea3975ed111117712d353ad4b2d98))
* **references:** record the completed ViT-B strategy triplet ([7beed4e](https://github.com/f6ra07nk14/structcast-model/commit/7beed4e65be4f334911fc2f96d33118885d253c5))
* **references:** replace concrete host paths with placeholders ([9fb54a9](https://github.com/f6ra07nk14/structcast-model/commit/9fb54a9418ddb402fe4453cedb5ea6b5b92adad9))
* **references:** restate the micro-benchmark method instead of keeping the scripts ([d517f04](https://github.com/f6ra07nk14/structcast-model/commit/d517f049ea6754bb3a45ed1bd567b1807b1ec70d))
* **references:** state the vision augmentation recipe ([bdd1056](https://github.com/f6ra07nk14/structcast-model/commit/bdd10567874956e9286c09b18452423f9f515b6e))
* rewrite README and reference docs for the Learner design ([b7f08c0](https://github.com/f6ra07nk14/structcast-model/commit/b7f08c09b6739edee25cdffb131fc9190ec437bb))
* update worktree guidelines for devcontainer compatibility ([e97ac03](https://github.com/f6ra07nk14/structcast-model/commit/e97ac035dd25dceea2c2fcb89ffa64829582631d))


### 📝 Examples

* add simple training tutorial ([76bf107](https://github.com/f6ra07nk14/structcast-model/commit/76bf107b5d20899ff35ea0016723b76539ed11d4))
* **torch:** decoder-only transformer LM on Tiny Shakespeare ([0148295](https://github.com/f6ra07nk14/structcast-model/commit/01482954a4e8962ca0743fa2bbc5cde3ebe50f32))
* **torch:** ViT-B/16 configuration bit-exact against timm ([d570dbc](https://github.com/f6ra07nk14/structcast-model/commit/d570dbc2899f464f57444324d669a003e72d23a5))


### 💎 Features

* **builders:** add INPUT_SHAPES tensor specification with dtype and initializer support ([323c664](https://github.com/f6ra07nk14/structcast-model/commit/323c6640e4b3c2f9dfb4e7d53169574ae0f77dfc))
* **commands:** compile per shard block and slim train() into orchestration ([fdc6949](https://github.com/f6ra07nk14/structcast-model/commit/fdc69496d2cd299803c9d747646f70a4958c3a8c))
* **devcontainer:** add GitHub CLI feature to devcontainer configuration ([a4ea145](https://github.com/f6ra07nk14/structcast-model/commit/a4ea145231b31871d1723c7b3fff281b133544ab))
* **torch:** add CausalSelfAttention layer ([2b1dc8d](https://github.com/f6ra07nk14/structcast-model/commit/2b1dc8dacdb843195b063d992bce390b9e38fd7f))
* **torch:** add distributed strategies owning wrap, weight sync, and checkpoint state ([a32e964](https://github.com/f6ra07nk14/structcast-model/commit/a32e9640253c219eaa30148efede782f49b4042a))
* **torch:** shard FSDP2 models per block via shard_modules patterns ([f20438b](https://github.com/f6ra07nk14/structcast-model/commit/f20438bb60b5ceb5d3f0146cfdc5f9be86a80b04))
* **torch:** strategies own compilation placement; distributed module leaves the lazy shim ([cae3998](https://github.com/f6ra07nk14/structcast-model/commit/cae3998e5f7e8b0fd10f4002f0ab596db7c4d729))


### 🔧 Fixes

* address adversarial review findings across the redesign ([52041f0](https://github.com/f6ra07nk14/structcast-model/commit/52041f02c143b736ad80356dbeea70a678405a80))
* **base-trainer:** make DataProvider not runtime_checkable for Python 3.11 ([06bceff](https://github.com/f6ra07nk14/structcast-model/commit/06bceffb88f21bbb0e42176767709a16a881a708)), closes [#121](https://github.com/f6ra07nk14/structcast-model/issues/121)
* **builders:** report unscaled loss when accumulating gradients ([16332ae](https://github.com/f6ra07nk14/structcast-model/commit/16332ae0f5d8b4563741ffbf31915b80891ede7d))
* **cli:** build TimmDataProvider for timm dataset wrappers in train ([1c11432](https://github.com/f6ra07nk14/structcast-model/commit/1c114321cc913803e0d8a005307bd8271bd6a634))
* **commands:** assign the display callbacks instead of augmenting a Sequence ([158c255](https://github.com/f6ra07nk14/structcast-model/commit/158c25509955a9df510155425d016405c3fc7407))
* **dependencies:** migrate to structcast 2.0 security API ([a4548cf](https://github.com/f6ra07nk14/structcast-model/commit/a4548cf4c8b2aadf4fdff1b53a8202e396b5370b))
* **dependencies:** update structcast version to 2.0.0 ([c4a7a45](https://github.com/f6ra07nk14/structcast-model/commit/c4a7a45edbd72e5dc1d221d993f4d3cc20b67ba6))
* **dependencies:** update structcast version to 2.0.0 ([0a7b50d](https://github.com/f6ra07nk14/structcast-model/commit/0a7b50dfc662bedbd93f07cfebf3996931f7593d))
* **torch:** route proxy optimizers through their own state dicts and make python -m work ([1dc4e21](https://github.com/f6ra07nk14/structcast-model/commit/1dc4e216659617c5509b5096e97a644817cc4705))
* **torch:** set sync flags for backward-time readers and respect user-frozen parameters ([bfb8b2d](https://github.com/f6ra07nk14/structcast-model/commit/bfb8b2d7d63984e2f963c654b6df148af015af3b))


### 🚀 Performance

* **torch:** compile flow functions only on a single device, backed by H200 step timings ([f77510a](https://github.com/f6ra07nk14/structcast-model/commit/f77510af5b416a1bd62c5b70b92dbaa1f8f4640b))


### 🔨 Refactor

* **commands:** drop configure_security calls made obsolete by structcast 2.0 ([b938ad0](https://github.com/f6ra07nk14/structcast-model/commit/b938ad0b302be92f32435967341717c996b4eee5))
* **commands:** route all compilation through strategy.compile ([35e36b9](https://github.com/f6ra07nk14/structcast-model/commit/35e36b917fe46c16bc19919aabd9c7c88b44485e))
* remove InferenceWrapper from __all__ exports ([9f91516](https://github.com/f6ra07nk14/structcast-model/commit/9f9151636b15f50b43a09b18c17baf74e936b273))
* remove InferenceWrapper protocol definition ([3baa7a2](https://github.com/f6ra07nk14/structcast-model/commit/3baa7a2ae5332e1c51392390ece5f24a9f44183d))
* **torch:** drop the redundant is_successful guard around _imports.check() ([bfbb150](https://github.com/f6ra07nk14/structcast-model/commit/bfbb150abb435cc832e439f535890c091748662f))
* **torch:** move distributed env init into the strategy module via try_import guards ([ea804cb](https://github.com/f6ra07nk14/structcast-model/commit/ea804cbd4fb6bff5e709f653979aa4cfa2965abe))
* **torch:** replace the None logger with a NullLogger null object ([77cfc87](https://github.com/f6ra07nk14/structcast-model/commit/77cfc87d213eeafa7d4daf8d79cdd7096037d163))
* **torch:** sync_gate becomes a plain setter statement ([d3d5db7](https://github.com/f6ra07nk14/structcast-model/commit/d3d5db7bd779c1729eb6918186aa378356199a6f))


### ✨ Style

* satisfy ruff line length and zip strictness across the new codegen ([f3ec878](https://github.com/f6ra07nk14/structcast-model/commit/f3ec878fbbe71b14adc4e0309c0e4b6219218572))


### 🚨 Tests

* **torch:** prove distributed semantics with analytic two-rank tests ([d40a6f0](https://github.com/f6ra07nk14/structcast-model/commit/d40a6f0469a8d93c8742f5ad99d3331e582be32b))
* **torch:** prove native-engine weight decay and layer decay end to end ([a94bf0e](https://github.com/f6ra07nk14/structcast-model/commit/a94bf0e5b3340b52d368418573f8f62d2afd10dd))

## [2.0.0](https://github.com/f6ra07nk14/structcast-model/compare/v1.5.0...v2.0.0) (2026-04-12)


### 💥 Breaking Changes

* restructure backward template schema for multi-optimizer GAN training support ([26f0dd2](https://github.com/f6ra07nk14/structcast-model/commit/26f0dd2c63b5543b0c56fbed7a4d884bc0b7f977))

## [1.5.0](https://github.com/f6ra07nk14/structcast-model/compare/v1.4.0...v1.5.0) (2026-04-09)


### 👷 Build

* add all-cpu dependency for unified installation of JAX, Torch, and TensorFlow support ([de187f0](https://github.com/f6ra07nk14/structcast-model/commit/de187f0c45a3add66515bc994c62fbc0cb932902))
* add support for additional structcast-model configurations for Keras and CUDA ([ba20716](https://github.com/f6ra07nk14/structcast-model/commit/ba20716a8eb5bad7c6d171c2da2c54db3aef5f93))
* update Keras dependencies for Torch integration and enhance all-cpu configuration ([50e0c81](https://github.com/f6ra07nk14/structcast-model/commit/50e0c8195a116ee88874fbfc74d84788d8e0daf6))


### 📦 Other

* correct flow structure and output definitions in ConvNeXtV2 YAML configurations ([6cf1284](https://github.com/f6ra07nk14/structcast-model/commit/6cf1284bbaf267fec41e6af69f538d8e67979f28))
* reorganize configuration files under cfg/torch/ directory ([dab047f](https://github.com/f6ra07nk14/structcast-model/commit/dab047f229e6a5a67c6d8f6c2aa5ff45929983bc))
* update Block flow to use string expression for addition operation ([398b0e4](https://github.com/f6ra07nk14/structcast-model/commit/398b0e4305f6710c2a80dbcd41960b85f9d9cdf1))


### 🦊 CI/CD

* update extras in tox.ini to replace 'torch-cpu' with 'all-cpu' ([2cc6c0b](https://github.com/f6ra07nk14/structcast-model/commit/2cc6c0ba7f13f8ba897b2e3a379dc706b36887e2))


### 📔 Docs

* add clarification on avoiding unnecessary noqa and type ignore comments ([264805a](https://github.com/f6ra07nk14/structcast-model/commit/264805ae1223be0c1ba67400681d06078481fb9d))
* add Schema Reference section to README.md ([3e686b3](https://github.com/f6ra07nk14/structcast-model/commit/3e686b3122701acaf9dacf203496ce62013a298d))
* update README files to reflect multi-framework support for PyTorch, Flax, and Keras ([499cdf9](https://github.com/f6ra07nk14/structcast-model/commit/499cdf932e57e971cddf7cce1703cb37b05be7e1))


### 💎 Features

* add class method to get class instance with rngs and training parameters ([99397b0](https://github.com/f6ra07nk14/structcast-model/commit/99397b013f6562ab7a718d992849cf5f4ce0b889))
* add command to measure average inference time of Flax models ([7aeb480](https://github.com/f6ra07nk14/structcast-model/commit/7aeb480d1133e700fca21a1183910ff01c53c7a1))
* add functions to retrieve available JAX devices and specific device by string ([a1d7e38](https://github.com/f6ra07nk14/structcast-model/commit/a1d7e38f60e6d054530984ba66154b0b43b133da))
* add get_keras_device function to retrieve available Keras devices ([164825d](https://github.com/f6ra07nk14/structcast-model/commit/164825dce3fcba308ec6e490de575cc09b738a07))
* add inference time measurement command and compile option for PyTorch models ([c1881c6](https://github.com/f6ra07nk14/structcast-model/commit/c1881c63c45bdc7bdfe222e2ba9b2a97f2ceac44))
* add inference time measurement command for Keras models with compile options ([14a8ae3](https://github.com/f6ra07nk14/structcast-model/commit/14a8ae369ecf81fe6d038d30f9ba3f7e0fe279c0))
* add Keras module and Global Response Normalization layer implementation ([59b9711](https://github.com/f6ra07nk14/structcast-model/commit/59b9711b9f69e0988611ff32936b00f13abdaf29))
* add keras to module exports and import structure ([9daba40](https://github.com/f6ra07nk14/structcast-model/commit/9daba40f4d0bdf3c62d488d790c3b4c050e79ed1))
* add KerasBuilder, FlaxBuilder, CLI commands, and ConvNeXtV2 YAML configs ([ae3cecc](https://github.com/f6ra07nk14/structcast-model/commit/ae3cecc8c4e5c5292fb8d74a25910f27dd6d8eb8))
* add matmul_precision option and warmup runs for inference time measurement ([4b2517e](https://github.com/f6ra07nk14/structcast-model/commit/4b2517e81e2a76e6bdfc414a4e8a876728f99b40))
* add trainer helpers for Keras models with input creation functions ([737a11c](https://github.com/f6ra07nk14/structcast-model/commit/737a11c574788f6fe3503ba858878c10f3ff6476))
* add warmup runs for inference time measurement ([df24075](https://github.com/f6ra07nk14/structcast-model/commit/df2407595fd8bc878f326b72863b442d15bb9865))
* implement Global Response Normalization (GRN) layer in Flax ([548d8ea](https://github.com/f6ra07nk14/structcast-model/commit/548d8ea9ead60df528a9c5524529d8447ca240e6))


### 🔧 Fixes

* add training parameter to FlaxLayerIntermediate constructor and call method ([662ee94](https://github.com/f6ra07nk14/structcast-model/commit/662ee9469e5a48e1291118996c0bc93261dad2dd))
* add TypeAlias annotation for _Intermediate in test_intermediate_get_scripts ([c7b8626](https://github.com/f6ra07nk14/structcast-model/commit/c7b862666ee46ff1a9145ecbf1c48c5271f3f94d))
* address code review - list type annotation, drop_path RNG seed, and shape comment ([401ec89](https://github.com/f6ra07nk14/structcast-model/commit/401ec8967c2cc62d92d5c98a090571cac2d676df))
* correct instantiation method for compile pattern in measure_inference_time function ([49474f7](https://github.com/f6ra07nk14/structcast-model/commit/49474f770ddaf1f1d579e9dee49dc2ff42bb6d61))
* correct variable name from 'warnup_runs' to 'warmup_runs' in cmd_flax.py, cmd_keras.py, and cmd_torch.py ([13172f0](https://github.com/f6ra07nk14/structcast-model/commit/13172f0ad23397197311919574f8cd57a64d1c43))
* expose get_jax_device and get_jax_devices in module exports ([67d8665](https://github.com/f6ra07nk14/structcast-model/commit/67d8665b95b38eeedc5d18de057b5cabe9c8dc36))
* expose GlobalResponseNorm in module exports ([fde565b](https://github.com/f6ra07nk14/structcast-model/commit/fde565b6bf2b972bcb6237cfe23e8b145f47b6b7))
* expose GlobalResponseNorm in module exports ([efef061](https://github.com/f6ra07nk14/structcast-model/commit/efef0610da07cc1824592a617c0b0213369c89fd))
* update compile pattern check to ensure it's not None before compilation ([3d0bb8e](https://github.com/f6ra07nk14/structcast-model/commit/3d0bb8e6fbe955d0e1c99a6beac4b4c9f1ee0ce1))
* update default imports to include flax.nnx module ([9089c81](https://github.com/f6ra07nk14/structcast-model/commit/9089c815d951590a7a5ee8db4f1f45a519bc39de))
* update dockerfile to use all-cpu dependency for consistent environment setup ([81c34ab](https://github.com/f6ra07nk14/structcast-model/commit/81c34ab6884da40a52181aec226432784756590a))
* update GlobalResponseNormalization to GlobalResponseNorm in ConvNeXtV2.yaml ([1813014](https://github.com/f6ra07nk14/structcast-model/commit/1813014408afec4e26c20edd332b5a494f8ceb2a))
* update GlobalResponseNormalization to GlobalResponseNorm in ConvNeXtV2.yaml ([f8852ee](https://github.com/f6ra07nk14/structcast-model/commit/f8852ee1a37012168a21fd457a049dfb56431928))
* update import for GlobalResponseNormalization in layers module ([f9f9ea1](https://github.com/f6ra07nk14/structcast-model/commit/f9f9ea1e7f0786647130e08f551a58b9c91d0315))
* update jax import handling and instantiate compile pattern in measure_inference_time function ([60cedc4](https://github.com/f6ra07nk14/structcast-model/commit/60cedc4125b6044ddf0e5f20f63bbaf43c8c249c))
* update keras dependency to version 3.13.2 in pyproject.toml ([c78d40d](https://github.com/f6ra07nk14/structcast-model/commit/c78d40d6e9b55d1292d8ebc2efafe0f47bf18f6f))
* update model compilation to use instantiator for compile pattern ([0a9e4b9](https://github.com/f6ra07nk14/structcast-model/commit/0a9e4b973e5675644028bd8b11713be40de1e97e))


### 🔨 Refactor

* remove unused Add, Multiply, ReduceSum, and ScaleIdentity layers ([04c14d0](https://github.com/f6ra07nk14/structcast-model/commit/04c14d0a5dec64fb2d28f96a4ea434876beeb0fa))
* remove unused Flax layer classes and update module exports ([52d7b96](https://github.com/f6ra07nk14/structcast-model/commit/52d7b9665b2c8f1c417bda24443498a33e588ab9))
* rename instantiate function to instantiate_object for clarity ([29791c1](https://github.com/f6ra07nk14/structcast-model/commit/29791c1fbf1848851467a064bf734bfb7aca77c9))
* replace _instantiate with instantiate in cmd_torch and update utils with detailed docstrings ([dd60a05](https://github.com/f6ra07nk14/structcast-model/commit/dd60a05dd0534a4ac24b66df4d3d75ddf3604dc5))
* replace DropPath layer with Dropout layer in ConvNeXtV2.yaml ([7a68465](https://github.com/f6ra07nk14/structcast-model/commit/7a6846562c3237a32ea3007934eac872ae970cc7))
* simplify _forward_flow method in KerasLayerIntermediate ([fcf3863](https://github.com/f6ra07nk14/structcast-model/commit/fcf386346f396174d0e3aaf3b63a829de19cdd0f))
* simplify import statements and improve docstring formatting in flax_builder.py ([3982f78](https://github.com/f6ra07nk14/structcast-model/commit/3982f781a7cd926178c6a009041757b30b087c47))
* simplify training parameter handling in Flax and Keras model call methods ([449ae01](https://github.com/f6ra07nk14/structcast-model/commit/449ae01d35c970c061dd6d92c9b13ff4bf65a4c8))
* streamline GlobalResponseNorm class by removing redundant imports and simplifying parameter initialization ([087e514](https://github.com/f6ra07nk14/structcast-model/commit/087e514802eeab1cb0d8e6fb9fe20415e13d6aaf))
* streamline import statements and improve formatting in keras_builder.py ([4a555fb](https://github.com/f6ra07nk14/structcast-model/commit/4a555fbbb26da5fa9df025defd16d216b184a4f5))
* update configuration paths for ConvNeXtV2 models to include 'torch' directory ([b68b762](https://github.com/f6ra07nk14/structcast-model/commit/b68b762081250ef24def5ec3fe47392b9693fe99))
* update configuration paths for Flax and Keras ConvNeXt models to include respective directories ([a1d66ec](https://github.com/f6ra07nk14/structcast-model/commit/a1d66ec843cfcab61ac04ff08f48ce191446d7d0))
* update ConvNeXtV2.yaml structure and layer configurations ([a745980](https://github.com/f6ra07nk14/structcast-model/commit/a74598072362b259e544d0739e9b534324baaec4))
* update ConvNeXtV2.yaml to enhance layer configurations and streamline flow definitions ([2a97154](https://github.com/f6ra07nk14/structcast-model/commit/2a971549f85b88797707ff0ca06bf8b98fe8843f))
* update type annotations for ReinMaxCore methods to use FunctionCtx ([69fd891](https://github.com/f6ra07nk14/structcast-model/commit/69fd8918e1a2d4caed67f0953d4d040c285ec0e1))
* update type check for shape parameter in create_torch_inputs function ([10526ad](https://github.com/f6ra07nk14/structcast-model/commit/10526adc0ab04a865e923999e27b6497f3a3d384))
* update YAML structure for Backbone, Stem, DownSample, Block, LayerNorm, and DropPath layers ([0191737](https://github.com/f6ra07nk14/structcast-model/commit/0191737d652450210107b1151f0e95a1b882a73a))


### ✨ Style

* add type hint for kwargs parameter in GlobalResponseNormalization initializer ([0c6348d](https://github.com/f6ra07nk14/structcast-model/commit/0c6348d4d81312a6509986cf23352c9e6270cb40))
* remove unnecessary noqa comment in test_intermediate_get_scripts_raises_not_implemented ([672aed2](https://github.com/f6ra07nk14/structcast-model/commit/672aed225520ced014bf8f02611d9c3479314bc8))


### 🚨 Tests

* add unit tests for cmd_flax command structure and help functionality ([5c28193](https://github.com/f6ra07nk14/structcast-model/commit/5c28193dc4647bb63ef495ff968287d2eb02de2e))
* add unit tests for cmd_keras command structure and help functionality ([9bac46b](https://github.com/f6ra07nk14/structcast-model/commit/9bac46b398a361947562222be582c25e6d611a94))
* update dependency key from 'instantiate' to 'instantiate_object' in test_cmd_torch.py ([32078c5](https://github.com/f6ra07nk14/structcast-model/commit/32078c50416021fc13abb6d122e1039434e2046c))
* update import assertion for FlaxBuilder to include 'flax.nnx' ([b7aa904](https://github.com/f6ra07nk14/structcast-model/commit/b7aa90468746ca653de8708e3666041e5a006bc7))

## [1.4.0](https://github.com/f6ra07nk14/structcast-model/compare/v1.3.0...v1.4.0) (2026-04-01)


### 👷 Build

* upgrade package dependencies ([63590e9](https://github.com/f6ra07nk14/structcast-model/commit/63590e9c8553a338153a378e5452651b24c703b2))


### 📦 Other

* add data_file configuration for pytest coverage ([2b8d14b](https://github.com/f6ra07nk14/structcast-model/commit/2b8d14b8a4ac7c1a43d15432de566f8e0fc33719))


### 💎 Features

* add trainer pattern option for model training configuration ([af8da80](https://github.com/f6ra07nk14/structcast-model/commit/af8da8061ac404147723716d21b90c0405dd0c4a))


### 🔧 Fixes

* correct elapsed_time calculation in training and validation steps ([857a637](https://github.com/f6ra07nk14/structcast-model/commit/857a6374a1dea4af7841947343d95f63cf12bb8f))
* correct forward method input handling in TorchLayerIntermediate ([29417b9](https://github.com/f6ra07nk14/structcast-model/commit/29417b95533a689dd5731a2c72862198202ebd48))
* reorder output attribute check in _get_module_outputs function ([7e1c838](https://github.com/f6ra07nk14/structcast-model/commit/7e1c838439d367ef6a0baeebbfe2151d786651e3))

## [1.3.0](https://github.com/f6ra07nk14/structcast-model/compare/v1.2.0...v1.3.0) (2026-03-28)


### 👷 Build

* add mlflow and flops as extra dependencies in Dockerfile ([7fa3847](https://github.com/f6ra07nk14/structcast-model/commit/7fa3847fe320112abd1a28339ecafbee0390e4da))


### 🦊 CI/CD

* add coverage configuration for testing with multiprocessing support ([3d2348d](https://github.com/f6ra07nk14/structcast-model/commit/3d2348d972b01cbcf11b7d10c111e10afaa2b203))
* add mlflow and flops dependencies to tox configuration ([ab087be](https://github.com/f6ra07nk14/structcast-model/commit/ab087be0c138533ae832c122f2a464647ff0bcc5))
* update tox command to run without parallel execution ([85c0036](https://github.com/f6ra07nk14/structcast-model/commit/85c00366d207d7f06e35d4435e919ff79e491682))


### 💎 Features

* add initializer patterns option for model instantiation in train function ([75ec398](https://github.com/f6ra07nk14/structcast-model/commit/75ec3981346a3bff9c2181087aa7792e6f53fa45))
* add options for training and validation step patterns in train function ([eeb8cdb](https://github.com/f6ra07nk14/structcast-model/commit/eeb8cdb62a8f7039d01378f94b9939d9f3ca0591))


### 🔧 Fixes

* add runtime_checkable decorator to protocol classes for enhanced type checking ([d4bb823](https://github.com/f6ra07nk14/structcast-model/commit/d4bb823f67de546a117b70235717259a437878e3))


### 🔨 Refactor

* streamline training and validation step instantiation in train function ([606cca4](https://github.com/f6ra07nk14/structcast-model/commit/606cca42b0a8eb9cff32942c3f0eb20f41adaf99))

## [1.2.0](https://github.com/f6ra07nk14/structcast-model/compare/v1.1.0...v1.2.0) (2026-03-17)


### 🦊 CI/CD

* remove unused 'flops' extra from tox configuration ([ab8223a](https://github.com/f6ra07nk14/structcast-model/commit/ab8223a39a868ed310d1e993ec484e8f9ef5cebf))


### 📔 Docs

* add parallel check command to README ([347cc3c](https://github.com/f6ra07nk14/structcast-model/commit/347cc3ce19d562d320971e97bf28e05a51f5fee1))
* add warning about SyncBatchNorm for multi-GPU training with DDP ([e0471b7](https://github.com/f6ra07nk14/structcast-model/commit/e0471b73bb11a615132742f2bcd1ff1d77084368))
* enhance documentation for distributed training support with torchrun ([13aea9d](https://github.com/f6ra07nk14/structcast-model/commit/13aea9d45c1c4e3fa9e19fb7d43a91d2cf5d7242))


### 💎 Features

* enhance BestCriterion and TorchTracker for improved tracking and callback functionality ([6692427](https://github.com/f6ra07nk14/structcast-model/commit/6692427c8ad09952a7ad5a336fcef8ba7972ba54))
* enhance TimmDataLoaderWrapper with distributed support and dataset management ([a90ebff](https://github.com/f6ra07nk14/structcast-model/commit/a90ebff348031aa6b8849bac629dad8039a15743))
* refactor training logic and enhance distributed training support ([b8e8a74](https://github.com/f6ra07nk14/structcast-model/commit/b8e8a740b542a67fb9986041c0d3141e3daa3728))


### 🔧 Fixes

* enhance TimmDataLoaderWrapper for distributed training support with additional parameters ([305ae08](https://github.com/f6ra07nk14/structcast-model/commit/305ae08415c1eec8434f596916674f4ffea4ef6a))
* fix the bugs of returning incorrect update code script in TorchBackwardIntermediate ([89dffbe](https://github.com/f6ra07nk14/structcast-model/commit/89dffbe9bffd260239a71da65f13ab7ef93df125))
* update command-line argument flags for consistency in training options ([35ea5a9](https://github.com/f6ra07nk14/structcast-model/commit/35ea5a906fd67587c2f85189684f26c78bb02e7a))


### ✨ Style

* update noqa comments in train function for improved linting compliance ([f14806e](https://github.com/f6ra07nk14/structcast-model/commit/f14806ec3c2ca000074c7c193cdfdd32f9844c24))

## [1.1.0](https://github.com/f6ra07nk14/structcast-model/compare/v1.0.0...v1.1.0) (2026-03-12)


### 👷 Build

* remove unnecessary build-essential installation from Dockerfile ([da8bb18](https://github.com/f6ra07nk14/structcast-model/commit/da8bb1884f1b49a6029a050a63d5d81f2f1b94b6))
* specify version constraints for torch dependencies in pyproject.toml ([d639062](https://github.com/f6ra07nk14/structcast-model/commit/d6390620dee0c36374bd6bdcd14209529ece19ea))


### 📦 Other

* add notes on optional dependencies for JAX, TensorFlow, and Keras ([24807f4](https://github.com/f6ra07nk14/structcast-model/commit/24807f48d5148026a09c4e82a2f02049869ce049))
* increase number of workers to improve data loading performance ([e018367](https://github.com/f6ra07nk14/structcast-model/commit/e018367e60b3156506984b53f15f7a38cef176ec))


### 📔 Docs

* Remove `from __future__ import annotations` requirement from code style ([bc7f99d](https://github.com/f6ra07nk14/structcast-model/commit/bc7f99d7a4bf255cfdedbc8b0ac37dffb2d6eb3c))


### 💎 Features

* add context manager for callback session management and refactor invoke_callback signature ([5a7b993](https://github.com/f6ra07nk14/structcast-model/commit/5a7b9939009af53764a7a819de51658f99be124d))
* implement NamedCallbackList for enhanced callback management and refactor callback registration in trainer and optimizer ([c5d6e58](https://github.com/f6ra07nk14/structcast-model/commit/c5d6e58addc8087619f7b2c9bc1786df893d1226))


### 🔧 Fixes

* add __all__ exports to various layers for improved module visibility ([1a207d1](https://github.com/f6ra07nk14/structcast-model/commit/1a207d12b43456f40b65b30d758530a2fa8819d6))
* enhance error handling in format_template for undefined jinja2 variables ([85bdcb6](https://github.com/f6ra07nk14/structcast-model/commit/85bdcb691f0ea30b0586c1f33ce73e2a88622df3))


### 🔨 Refactor

* capture narrowed reference for metrics_tracker in TorchTracker ([ea80766](https://github.com/f6ra07nk14/structcast-model/commit/ea80766a9bbb50a65410d5138ce902f198d74439))

## 1.0.0 (2026-03-11)


### 👷 Build

* add dependencies for ptflops and calflops in pyproject.toml ([5bda857](https://github.com/f6ra07nk14/structcast-model/commit/5bda85730cabce49693423d08273b52316675589))
* add dev Dockerfile ([f1dd202](https://github.com/f6ra07nk14/structcast-model/commit/f1dd202577cf75d59acb8bf0d976afc37066b68e))
* add extra dependencies for torch-cpu and flops in Dockerfile ([85c2891](https://github.com/f6ra07nk14/structcast-model/commit/85c2891432822a5acb9e0a264a47c85a714c1b83))
* add optional dependencies for various JAX, TensorFlow, and PyTorch configurations ([2be9c8f](https://github.com/f6ra07nk14/structcast-model/commit/2be9c8f0dd71a3bfaa62440eb707c0d3065770f2))
* add pytest command to type-checking environment and remove all-checks environment ([6ba5487](https://github.com/f6ra07nk14/structcast-model/commit/6ba5487c9ee61d611cbabaa6f3c8fc5b7c997d72))
* format dev dependencies in pyproject.toml for better readability ([2d77437](https://github.com/f6ra07nk14/structcast-model/commit/2d7743703db1830e3643d77064ce56b51f221b6d))
* Remove redundant extra dependency in uv sync command ([312f3db](https://github.com/f6ra07nk14/structcast-model/commit/312f3db615dba4cd685a009dc335dd4ae32a964f))
* Reorganize project URLs section in pyproject.toml for consistency ([eb85f85](https://github.com/f6ra07nk14/structcast-model/commit/eb85f857ced6e9382c4d08d8de8f855fc3dce395))
* update CI configuration and Dockerfile to use actions/checkout@v6 and remove Node.js installation ([f4b9436](https://github.com/f6ra07nk14/structcast-model/commit/f4b9436535ffdc4ef33ca397281c3655f7aacdf6))
* update dependencies in pyproject.toml and adjust structcast version ([0ccea61](https://github.com/f6ra07nk14/structcast-model/commit/0ccea6165721eb035750fdb667875448cee1b754))
* update numpy dependency to restrict version below 2.4.0 ([1507653](https://github.com/f6ra07nk14/structcast-model/commit/1507653ada6a85ee6a5554e6a5bac8b5bc415ec8))
* update package dependencies ([3a28e36](https://github.com/f6ra07nk14/structcast-model/commit/3a28e367d2fed5e3736e10ffba94e8696b49176d))
* Update project metadata in pyproject.toml for clarity and accuracy ([599f2ce](https://github.com/f6ra07nk14/structcast-model/commit/599f2cef727b345c0942d17cab0c05e3c02f82f1))
* update pydantic version to 2.12.5 in pyproject.toml and uv.lock ([45e4bd4](https://github.com/f6ra07nk14/structcast-model/commit/45e4bd4cfe0e8dbf822b19fdcaaaa8aa0e321226))
* update structcast source reference to latest commit ([179f141](https://github.com/f6ra07nk14/structcast-model/commit/179f1415aee039c3d60b86a8870c6a27726178d2))
* upgrade dependent packages ([78a45a4](https://github.com/f6ra07nk14/structcast-model/commit/78a45a4cfdcc0ff87ff049cd92dcec66c4ea4aa1))
* upgrade package dependencies ([55f795d](https://github.com/f6ra07nk14/structcast-model/commit/55f795dd284aa9daa083e385337bdde282446f51))
* upgrade package dependencies ([7214278](https://github.com/f6ra07nk14/structcast-model/commit/72142783f20ba58f387898d4705ee49bff7f4133))
* upgrade structcast package ([dc38be6](https://github.com/f6ra07nk14/structcast-model/commit/dc38be6cdbe8572dd0dc14bbc8ea011342f86749))


### 📦 Other

* add accumulate_gradients parameter to enhance gradient accumulation configuration ([645b3be](https://github.com/f6ra07nk14/structcast-model/commit/645b3beebdd44ce4b5c96a126d69c8a511350b97))
* add atto and cls configuration files for model parameters and flow definitions ([37fbfe1](https://github.com/f6ra07nk14/structcast-model/commit/37fbfe1d20a60253112919614dfd1aea950c3669))
* add configuration files for compile settings and EMA parameters ([234ed17](https://github.com/f6ra07nk14/structcast-model/commit/234ed17d368232e9c81b444befc1d90458669625))
* add configuration for mixup with prefetcher support in training pipeline ([db70109](https://github.com/f6ra07nk14/structcast-model/commit/db7010984290108dbf734e8e9195412587701b49))
* add ConvNeXtV2 configuration file with training parameters and backward pass settings ([23252cc](https://github.com/f6ra07nk14/structcast-model/commit/23252cc5bbf68bb9ddc14812c498ca7591d7686e))
* add ConvNextV2 model configuration with parameters and flow definitions ([dba39eb](https://github.com/f6ra07nk14/structcast-model/commit/dba39eb30a5c9798f8481c7abf1d09c170472eb5))
* add model_name parameter to DEFAULT and update LAYERS reference in BACKWARDS ([3b35acf](https://github.com/f6ra07nk14/structcast-model/commit/3b35acfa56705e15c8dae5ee94c3fc84546f0ea5))
* add soft classification loss configuration in soft_cls.yaml ([33f6a89](https://github.com/f6ra07nk14/structcast-model/commit/33f6a89699c75bee2646a3b99c252f4433dd656e))
* correct download parameter naming in mixup configuration ([442ae4f](https://github.com/f6ra07nk14/structcast-model/commit/442ae4f25985702362ff9ed2bc2ae11bb6973bd6))
* initialize project structure and add configuration files ([2f126ad](https://github.com/f6ra07nk14/structcast-model/commit/2f126adad2d0234380ad076b9a3e87a875c72d09))
* rename filename of ConvNeXtV2 configuration as "ConvNeXtV2" ([2b0d54b](https://github.com/f6ra07nk14/structcast-model/commit/2b0d54b988c04d715179ebce9c1225636e3db7bd))
* update ConvNeXtV2 configuration to enhance gradient clipping and layer decay settings ([9857162](https://github.com/f6ra07nk14/structcast-model/commit/9857162b8f886cf9d152e539ce764fc2ae92b1b8))
* update package versions for filelock and platformdirs, and bump structcast version to 1.1.1 ([c713beb](https://github.com/f6ra07nk14/structcast-model/commit/c713beba162b8d1782701f64dcc0298e82891681))


### 🦊 CI/CD

* consolidate dependencies in tox.ini for better management ([517a36c](https://github.com/f6ra07nk14/structcast-model/commit/517a36c633b0bf50c99e4d788a2e461b9084e544))
* Restore and implement Publish to PyPI job in CI pipeline ([81b5a3d](https://github.com/f6ra07nk14/structcast-model/commit/81b5a3dbd2779c7d1ff705cad81941d93214d0eb))


### 📔 Docs

* Add development environment setup instructions to README ([d67aa73](https://github.com/f6ra07nk14/structcast-model/commit/d67aa73a602b4f1820012c61a809386d3f6a50ff))
* Enhance documentation for StructCast-Model ([764b903](https://github.com/f6ra07nk14/structcast-model/commit/764b903778468e4a9a76326905b4f84e31b52732))
* Expand Copilot instructions with architecture, build & test, code style, testing conventions, and general conventions ([bcc7a88](https://github.com/f6ra07nk14/structcast-model/commit/bcc7a88d803c38fd08d0b299682a8e5d29e04aab))
* Revise README and SKILL documentation for clarity and consistency ([e1baaaa](https://github.com/f6ra07nk14/structcast-model/commit/e1baaaa95d6ee6bfe1eccc0439429412cc1f236b))
* Update README to enhance structure and navigation with new sections ([8ead274](https://github.com/f6ra07nk14/structcast-model/commit/8ead274d8ab7cac485378568f0d08665d0cd4ecd))


### 💎 Features

* add __call__ method to LayerIntermediate for saving scripts to a specified path ([75367aa](https://github.com/f6ra07nk14/structcast-model/commit/75367aa6551feb07b48e6afa700cbb4904856089))
* add __dir__ function and update __all__ in auto_name.py ([679f8f9](https://github.com/f6ra07nk14/structcast-model/commit/679f8f954195f14a1e62fc563b5e34a30ec7bbc5))
* add accuracy metrics for sparse categorical and top-k accuracy ([12fd73c](https://github.com/f6ra07nk14/structcast-model/commit/12fd73c0cecb448465437b410b9729d141d64f68))
* add backbone configuration for ConvNeXtV2 model ([9871a05](https://github.com/f6ra07nk14/structcast-model/commit/9871a05b8f65548271e185f9a607c55cf7e854a2))
* add BaseTrainer class for model training with support for callbacks and evaluation ([8230f7b](https://github.com/f6ra07nk14/structcast-model/commit/8230f7b120cf12169d89721d5d2881fde2638601))
* add BestCriterion class for tracking and invoking callbacks on best criteria ([c745452](https://github.com/f6ra07nk14/structcast-model/commit/c745452e16166d0cc008613213d9e44396d9dfba))
* add commands for creating PyTorch models and backward classes from configuration files ([e1c72ec](https://github.com/f6ra07nk14/structcast-model/commit/e1c72ecc804d7ee444108e029defaa048d270792))
* add CriteriaTracker class for tracking multiple criteria in PyTorch ([a1dc4c7](https://github.com/f6ra07nk14/structcast-model/commit/a1dc4c7889f1b8cd94fe449001a9e990c0207bfc))
* add default configuration for Timm dataset and remove obsolete mixup configuration ([ca73f07](https://github.com/f6ra07nk14/structcast-model/commit/ca73f07b7cfc7e8c5fe327821829e8e7cf34a6f7))
* add device synchronization ([6a00db8](https://github.com/f6ra07nk14/structcast-model/commit/6a00db89a4132fdb0e19841b977ff9b78ca6f4ce))
* add factory methods to create model and backward builders from configuration file paths ([e2bbc33](https://github.com/f6ra07nk14/structcast-model/commit/e2bbc33d0e4fbb5beef1fffa3cf1d3f558857e16))
* add format command to CLI for template configuration with parameter support ([2df641f](https://github.com/f6ra07nk14/structcast-model/commit/2df641f0602438b8977501279b66e9ad47a1bbfd))
* add from_models class method to TimmEmaWrapper for EMA model creation ([41f322c](https://github.com/f6ra07nk14/structcast-model/commit/41f322cc5b28a0c1eab7aca818096d55d92f7aba))
* add get_default_dir utility to enhance directory management across modules ([ee8186b](https://github.com/f6ra07nk14/structcast-model/commit/ee8186b9e1bba5ffea5b8b5586ce798f37c759a0))
* add jinja_filters module and integrate cumsum filter into jinja configuration ([8b360d4](https://github.com/f6ra07nk14/structcast-model/commit/8b360d413186fc8f94e9f18193c2b08320540bef))
* add layer name validation for UserLayer, LayerBehavior, and BackwardBehavior classes ([e4cfd6f](https://github.com/f6ra07nk14/structcast-model/commit/e4cfd6f9592230610e59e0a59e370df98acd2bb1))
* add layers to __all__ for improved module exports in Torch extensions ([a6a989a](https://github.com/f6ra07nk14/structcast-model/commit/a6a989ab51860c1138f3a654ee1142bac45c0e22))
* add lazy import utilities for deferred module loading and error handling ([cb8c7bb](https://github.com/f6ra07nk14/structcast-model/commit/cb8c7bb68ca4acbf8a56b1fecebba458f00bd509))
* add learning rates and parameter group names properties to torch builder ([4e306df](https://github.com/f6ra07nk14/structcast-model/commit/4e306df59646aecbab9c2946c783e18ca1c04ccd))
* add mixed precision type support for backward layers and update related configurations ([20f4ee9](https://github.com/f6ra07nk14/structcast-model/commit/20f4ee9a560a332a956f9e108692e4ad1484e8a9))
* add mlflow dependency to pyproject.toml ([3a8e853](https://github.com/f6ra07nk14/structcast-model/commit/3a8e853235b54f64c3bc732581559064e8924351))
* add model serialization method and improve type annotations in LayerIntermediate and BaseBuilder ([b5ec0b3](https://github.com/f6ra07nk14/structcast-model/commit/b5ec0b3b9fe755acdb7143fadaf40e811b997fa6))
* add optimizer creation and scheduling functionality ([d6d4e41](https://github.com/f6ra07nk14/structcast-model/commit/d6d4e41e97484e41a25a1b05b6e7783782f1d960))
* add print_value function to output and return a value ([6c27dd1](https://github.com/f6ra07nk14/structcast-model/commit/6c27dd10504b99a16b5c4256ee36d883526d39de))
* add properties for optimizers and grad scalers in torch_builder for enhanced model management ([f78f6ef](https://github.com/f6ra07nk14/structcast-model/commit/f78f6ef6f40112f4c780da9d18997a4b37224404))
* add ptflops and calflops commands for model complexity analysis ([131d2ef](https://github.com/f6ra07nk14/structcast-model/commit/131d2efbcc3e4cb467c4cdc361d3dc7c7edfa56a))
* add raise_error function to handle error messaging in jinja filters ([bea3683](https://github.com/f6ra07nk14/structcast-model/commit/bea3683c4548a50e6aead3d89a85ce74d6f98c84))
* add string conversion functions to_snake, to_pascal, and to_camel ([4687e69](https://github.com/f6ra07nk14/structcast-model/commit/4687e6913a1b135c05c34b705ea47057555b37dc))
* add support for user-defined layers in model creation ([287d30c](https://github.com/f6ra07nk14/structcast-model/commit/287d30cf7ab994f24f3f067ee1e2b6624e70ce0c))
* add TimmDatasetWrapper and TimmDataLoaderWrapper for enhanced data loading in training pipeline ([2146852](https://github.com/f6ra07nk14/structcast-model/commit/2146852700faecefb09bc846ffae032320b2ee41))
* add TorchForward and TorchLogger classes for model forward pass and logging ([0ee32c8](https://github.com/f6ra07nk14/structcast-model/commit/0ee32c8bec267d0261c27013a11824bee1deb08d))
* add training and validation dataset pattern options to train function ([908c1f2](https://github.com/f6ra07nk14/structcast-model/commit/908c1f23f44c5c4cd6650bb12b6ca3fdc9f3dd62))
* add training command with model initialization and logging support ([55f9501](https://github.com/f6ra07nk14/structcast-model/commit/55f9501e7a80404be1155ba49a978cce560dde8d))
* add utility functions for creating torch inputs and determining device ([27bf314](https://github.com/f6ra07nk14/structcast-model/commit/27bf3144697d03afff1ceb2ac41b3a1fb46cfbc9))
* add utility functions for parsing YAML and reducing dictionaries ([1e0ea83](https://github.com/f6ra07nk14/structcast-model/commit/1e0ea8312bb2ab7e9fc820d41eea2dc8bd620044))
* add utility functions. ([b643b5b](https://github.com/f6ra07nk14/structcast-model/commit/b643b5bef873f58c277202b4099d54ba2de1a00d))
* add various layer implementations for PyTorch including Add, Multiply, Concatenate, and more ([12904b8](https://github.com/f6ra07nk14/structcast-model/commit/12904b89a793a9088b627bd5da1caa5e23d5317e))
* enhance cmd_torch.py with EMA support, model signature logging, and improved parameter handling ([a3db4ba](https://github.com/f6ra07nk14/structcast-model/commit/a3db4baa461cc99c7937693cc734d038d6e982b7))
* enhance configuration classes with extra fields and validation ([c80b86e](https://github.com/f6ra07nk14/structcast-model/commit/c80b86e605c7b2f5513e2574744634179a8d711d))
* enhance model initialization and add mixed precision support in training ([6bdfdbb](https://github.com/f6ra07nk14/structcast-model/commit/6bdfdbba41b15666bfb7fa4118d03cd4e68d08ca))
* enhance trainer with model handling and data loader wrappers ([644c2b9](https://github.com/f6ra07nk14/structcast-model/commit/644c2b966d3ca8c391422c46c38bd98ff3c6a1db))
* enhance UserDefinedLayer with imports validation and update LayerIntermediate imports type ([f83a7b5](https://github.com/f6ra07nk14/structcast-model/commit/f83a7b5b283939dea2fd7951bdb0068a57e0e6f5))
* implement backward layer handling with mixed precision support in TorchBackwardBuilder ([22b3824](https://github.com/f6ra07nk14/structcast-model/commit/22b3824e341fef973a9f5b3cb7b8f3f6c18bee4e))
* implement lazy loading and type checking for module imports in __init__.py ([2c956cf](https://github.com/f6ra07nk14/structcast-model/commit/2c956cfad6322ab3af65d007cbd62cf1f5eafa77))
* implement resolve_flow function and enhance UserDefinedLayer validation for inference flow ([f1500e6](https://github.com/f6ra07nk14/structcast-model/commit/f1500e64b5b5750633c5c3f4c6fabfd354d9e155))
* implement TorchLayerIntermediate and TorchBuilder for PyTorch model support ([ad13596](https://github.com/f6ra07nk14/structcast-model/commit/ad135961512d9fdb42fe8211f8b8bdcb42f67747))
* implement training state logging in epoch end for enhanced model tracking ([919501b](https://github.com/f6ra07nk14/structcast-model/commit/919501bf94f74ae69302310fef1d329e58e3f807))
* update __all__ exports and add initial_model and get_autocast functions in trainer ([5cbb122](https://github.com/f6ra07nk14/structcast-model/commit/5cbb1222efa70f8be34093339cefd2f2fdc242f8))
* update CLI application to StructCast Model and add PyTorch command support ([5dbe47a](https://github.com/f6ra07nk14/structcast-model/commit/5dbe47a86cbaafb356d263e1890dc1e143b51193))
* update InferenceWrapper protocol and enhance TimmEmaWrapper for better model handling ([783efe8](https://github.com/f6ra07nk14/structcast-model/commit/783efe8130fe21e09d03c1ce652496b490e8594d))


### 🔧 Fixes

* add __dict__ to the list of attributes in LazySelectedImporter ([a17a013](https://github.com/f6ra07nk14/structcast-model/commit/a17a01382bec98f6939d2261c2ae52b63f61a854))
* add cached_property for backward flow calculation in TorchBackwardIntermediate ([e906b0e](https://github.com/f6ra07nk14/structcast-model/commit/e906b0e38e4fde9d689e9446327cebbb22bb2315))
* add compile function parameter to TimmEmaWrapper for model compilation ([4988302](https://github.com/f6ra07nk14/structcast-model/commit/4988302c98d004559e30a44150995de303af55cf))
* add cross-device tracking for EMA models in TimmEmaWrapper ([9e2bf5f](https://github.com/f6ra07nk14/structcast-model/commit/9e2bf5f0a2ba62873deb3fe1e97ff434fec8a591))
* add logging for undefined validation step in BaseTrainer ([60cf11a](https://github.com/f6ra07nk14/structcast-model/commit/60cf11a81045bb34718fd0910bec9e3edf1678ed))
* add TimmDatasetWrapper to module exports ([cefafee](https://github.com/f6ra07nk14/structcast-model/commit/cefafee862ceeb8bcf164a82673782191fa56de0))
* adjust YAML structure for model validation and handle null samples correctly ([0956cb2](https://github.com/f6ra07nk14/structcast-model/commit/0956cb2b2a33d617e912d6b5e81d115bc11429e6))
* enhance circular reference detection in BaseBuilder and improve from_references handling ([94d08fc](https://github.com/f6ra07nk14/structcast-model/commit/94d08fc96e9cbcb0bd4eda358ccd85da45e42b91))
* handle empty parameters in backward class template formatting ([ce617ba](https://github.com/f6ra07nk14/structcast-model/commit/ce617ba34c458507aa257e89ba07c65fc7e783d1))
* handle None validation_step in BaseTrainer to prevent errors ([9c284db](https://github.com/f6ra07nk14/structcast-model/commit/9c284db876fd71be140f7a0e793f0d3d5eef8aa5))
* improve device type checking in TimmEmaWrapper for cross-device compatibility ([7f93d9c](https://github.com/f6ra07nk14/structcast-model/commit/7f93d9c572acf0bd05c0b7391c133a26dc54dbae))
* initialize inputs and outputs in Torch model constructor ([3d0bac5](https://github.com/f6ra07nk14/structcast-model/commit/3d0bac5c4d00cd31a38d6fc34e62636039157576))
* initialize outputs in the constructor of torch_builder ([8faf6e5](https://github.com/f6ra07nk14/structcast-model/commit/8faf6e51388dfc9ee0d8b59aca72b9e244a9590b))
* read file content before parsing JSON in load_json and load_any functions ([4d63938](https://github.com/f6ra07nk14/structcast-model/commit/4d63938a87dc942a41135986b98aabdb13a7265e))
* refactor BaseBuilder to use a dictionary for from_references and improve circular reference detection ([9790de0](https://github.com/f6ra07nk14/structcast-model/commit/9790de0cbf5196d5517aafe67e237b67c7262e09))
* remove commented-out warning and adjust dataclass configuration in BaseBuilder ([11bcd79](https://github.com/f6ra07nk14/structcast-model/commit/11bcd7926ecaf8077bb0ababff3b2d350d3c7316))
* remove default Parameters() in backward template instantiation ([6870420](https://github.com/f6ra07nk14/structcast-model/commit/687042063c7d1f712fca478f6082b8f79062caab))
* remove unused import and improve dataset size calculation logic ([4b465ad](https://github.com/f6ra07nk14/structcast-model/commit/4b465add3d356347c443ef1d3977aab9f29ee614))
* replace direct imports with torch namespace for consistency and clarity ([2cef19a](https://github.com/f6ra07nk14/structcast-model/commit/2cef19acb21683e58f6fbc05363858257f83bac0))
* replace load_yaml_from_string with path_or_any_parser in cmd_torch.py and add path_or_any_parser utility function ([c594489](https://github.com/f6ra07nk14/structcast-model/commit/c594489bc97fd053b15b3bbfad3076fec47d1ad7))
* set default label smoothing to 0.0 in cls.yaml ([b7a2559](https://github.com/f6ra07nk14/structcast-model/commit/b7a2559f8fa9c4b87636355f835e234a831a949b))
* simplify output resolution logic and enhance BaseBuilder initialization ([cbcb728](https://github.com/f6ra07nk14/structcast-model/commit/cbcb728f87836391eb418418aeff959327309158))
* simplify type alias definitions in types.py ([5c52939](https://github.com/f6ra07nk14/structcast-model/commit/5c529398325ca57de414a272fe6eb4cf81cb84a3))
* swap key and value to fix the issue of the dict output ([0d1b7a4](https://github.com/f6ra07nk14/structcast-model/commit/0d1b7a47b063f112691d9135e73ca9dc5535ade0))
* update backward flow logic to improve gradient accumulation handling ([ad27c60](https://github.com/f6ra07nk14/structcast-model/commit/ad27c605253fb2a7c1f2a91adaf36794c52e2a4b))
* update backward script assertions in TorchBackwardBuilder tests for accuracy ([46ec12a](https://github.com/f6ra07nk14/structcast-model/commit/46ec12ae128f7d4b67fea87055575c0d32f8c41f))
* update cross-device tracking logic in TimmEmaWrapper ([bbe4ff0](https://github.com/f6ra07nk14/structcast-model/commit/bbe4ff06f44045eab12cea60178290eee6b11a2b))
* update default seed value for reproducibility in training function ([495808f](https://github.com/f6ra07nk14/structcast-model/commit/495808f4e92c8a46ff6c90825e5095e22f9417c0))
* update format_template function to use reduce_dict directly and improve output handling ([5e68b3b](https://github.com/f6ra07nk14/structcast-model/commit/5e68b3bcab0d217bdca3c77595a61f717c5cace7))
* update get_dataset_size to use __len__ method for better compatibility with dataset objects ([03cb813](https://github.com/f6ra07nk14/structcast-model/commit/03cb813d357633ce3f5a99acd2543638d42df48b))
* update GLOBAL_CALLBACKS lambda functions to accept additional arguments for better compatibility ([9586ee2](https://github.com/f6ra07nk14/structcast-model/commit/9586ee21e14756539402bad66c459077394f913f))
* update import statement for SPEC_CONSTANT and improve error handling in _resolve function ([4996392](https://github.com/f6ra07nk14/structcast-model/commit/499639234fd0461aa19629d4a4f65ae6844cb453))
* update InferenceWrapper protocol to return Any type for model outputs ([2628e9a](https://github.com/f6ra07nk14/structcast-model/commit/2628e9a337967bf100cf36dbd0e184044cc7623f))
* update initial_model function to use Any type for raw inputs and outputs ([874742c](https://github.com/f6ra07nk14/structcast-model/commit/874742c08b2821198dd71155c81778bf7e60e0a9))
* update lambda functions in training callbacks to include index parameter for better clarity ([e38d82f](https://github.com/f6ra07nk14/structcast-model/commit/e38d82f2a3c8e00829e358cc633f6d0f93e6890b))
* update parameter types from Path to str for model and backward commands ([3339107](https://github.com/f6ra07nk14/structcast-model/commit/333910755e87ccdf5364ca2fbf993ab3ff50a54c))
* update parser functions to enhance YAML handling and rename for clarity ([f5d044c](https://github.com/f6ra07nk14/structcast-model/commit/f5d044cfa7893f2c3037c43c8799d0fe48f0e159))
* update template loading method and enhance raw data handling for WithExtra subclasses ([d9c76df](https://github.com/f6ra07nk14/structcast-model/commit/d9c76df326eb9a58fedfbb9a761b61703f21487f))
* update test cases to use model_validate for Serializable and Parameters, improving error handling and validation ([60dc65a](https://github.com/f6ra07nk14/structcast-model/commit/60dc65a46c7f932653ca968956a9e72110d15821))
* update train function to log model parameters as a dictionary ([cbb6b87](https://github.com/f6ra07nk14/structcast-model/commit/cbb6b87a7c14149d33bfd4c14b44379dc289426b))
* update type hints for compatibility with type checking ([66d561e](https://github.com/f6ra07nk14/structcast-model/commit/66d561ee9c8730d114880dcd16a1413eb7f4f879))


### 🔨 Refactor

* add default imports for PyTorch layers in TorchLayerIntermediate ([4f1dd99](https://github.com/f6ra07nk14/structcast-model/commit/4f1dd999957bcd467019bbe970d73deeb4fbbf69))
* add is_successful property to context manager for exception handling ([2e2ac0b](https://github.com/f6ra07nk14/structcast-model/commit/2e2ac0ba387aea4f71a2d67041716712f6805d2e))
* add Jinja filter for cumulative sum in schema.py ([781c2b7](https://github.com/f6ra07nk14/structcast-model/commit/781c2b75e9c3e598e3108cb4c7462362cc558867))
* add module_file option for model definition path in ptflops and calflops commands ([c65337c](https://github.com/f6ra07nk14/structcast-model/commit/c65337c2dd6ad71c3ac58593c89e278626ed020c))
* adjust code formatting for improved readability in base_builder.py ([d33984d](https://github.com/f6ra07nk14/structcast-model/commit/d33984d9d01c242f69fa7fbc4ff409de8b8cd2e5))
* enable global callbacks in base trainer and add CLI entry point ([0a440d4](https://github.com/f6ra07nk14/structcast-model/commit/0a440d4d1cc1fbee635fc60f73a0e484c463c198))
* enhance import collection and streamline layer configuration handling ([83e421a](https://github.com/f6ra07nk14/structcast-model/commit/83e421a3e3b355c0842d5889ab571c9e7a02a926))
* inline helper functions in train method for improved readability and maintainability ([e08a9d0](https://github.com/f6ra07nk14/structcast-model/commit/e08a9d0c3fd6ce00cd02540c1d0966d7e7051eda))
* integrate timm optimizer and scheduler with try import handling ([24628b6](https://github.com/f6ra07nk14/structcast-model/commit/24628b64d90461c932ae229461b199e9ca16e38a))
* introduce _to_pascal function for consistent PascalCase conversion ([30ebecd](https://github.com/f6ra07nk14/structcast-model/commit/30ebecdab4c4d82f8d6caa40108b5e5d1e9b265f))
* introduce BaseModelBuilder and TorchBackwardBuilder for enhanced model and backward layer building ([39186e0](https://github.com/f6ra07nk14/structcast-model/commit/39186e05cef2ee1a34a4dcae9e42aed0bef516d8))
* move global callback initialization to Callbacks class ([3859529](https://github.com/f6ra07nk14/structcast-model/commit/3859529b760f5a95f5e3cb85fe9330447c76dd7a))
* refactor field validators for optimizers and backwards in BackwardBehavior and UserDefinedBackward classes ([634ab2c](https://github.com/f6ra07nk14/structcast-model/commit/634ab2c5dd2b65bffb3c5f6a7fd4041925a35310))
* refactor the position of  channels_last and mixup_off_epoch fields in TimmDataLoaderWrapper ([e5390e1](https://github.com/f6ra07nk14/structcast-model/commit/e5390e11370d1bf9464d71f88eb32978c13c8f34))
* remove future annotations import and update type hints in base_builder and schema modules ([e785a53](https://github.com/f6ra07nk14/structcast-model/commit/e785a539231c1581a3a699cb45c003ea2e754bcb))
* remove try imports for timm modules in optimizers and trainer ([9f49efb](https://github.com/f6ra07nk14/structcast-model/commit/9f49efb41e1792171009a6e6622e4c55d18aefa3))
* remove TYPE_CHECKING imports and unnecessary __all__ declarations across multiple layer files ([a1b11e4](https://github.com/f6ra07nk14/structcast-model/commit/a1b11e4607ad38b534a574be39fd7a4611a26737))
* remove unused import and simplify class naming in BaseBuilder ([0f1bad6](https://github.com/f6ra07nk14/structcast-model/commit/0f1bad6c89cc4e065dc0a6f5e651061a6017c761))
* remove unused layer_call_name attributes from base and torch builders ([19c6517](https://github.com/f6ra07nk14/structcast-model/commit/19c65177e0f9fa25b588cfa60e3eaee17c809c6a))
* remove unused loader wrapper classes and clean up imports in trainer.py ([e9563a0](https://github.com/f6ra07nk14/structcast-model/commit/e9563a0d2d3040dc1a715f6a5498c01c424aebbf))
* remove unused variables in train function of cmd_torch.py ([d18e3c0](https://github.com/f6ra07nk14/structcast-model/commit/d18e3c08563633edabc204ab6c371494ba30cf15))
* rename input/output resolution functions and enhance docstrings for clarity ([e6be563](https://github.com/f6ra07nk14/structcast-model/commit/e6be56341b4d0588ce10785c831f0f7d9668f3b0))
* rename logger to tracker in BaseTrainer for clarity ([eee67b4](https://github.com/f6ra07nk14/structcast-model/commit/eee67b43c209390d11b442245dfc18683149bb26))
* rename TorchLogger to TorchTracker and update docstring for autocast ([e049c54](https://github.com/f6ra07nk14/structcast-model/commit/e049c54b9e8edfe41e8bf20702d65ada587dd0c2))
* replace OrderedDict with dict for classnames and backwards in LayerIntermediate and BaseBackwardBuilder ([fcac255](https://github.com/f6ra07nk14/structcast-model/commit/fcac2559850d2f77506e24ec02f0efda6dec5168))
* replace security imports with lazy imports for improved module loading ([8fed099](https://github.com/f6ra07nk14/structcast-model/commit/8fed0992fd1db3ebfa8fb6602786da27664bdb3c))
* replace security imports with lazy imports for improved module loading ([b3218be](https://github.com/f6ra07nk14/structcast-model/commit/b3218bee12ba64633eb894c3d3b7ae9aa4eceab2))
* simplify epoch end callbacks in train function for improved clarity and maintainability ([5ddc343](https://github.com/f6ra07nk14/structcast-model/commit/5ddc3430d1fbb165461dd86fce5dae901f7ddf8e))
* simplify parameter handling in _Template class ([e9932ef](https://github.com/f6ra07nk14/structcast-model/commit/e9932efb00d5e874f191d94a64c87a79487adcc5))
* simplify Parameters class by using "structcast.core.template.Parameters" as base class ([a0847bd](https://github.com/f6ra07nk14/structcast-model/commit/a0847bd43a7b042c7de9eb8dc63d61750c2130ab))
* simplify training function by removing unused code and enhancing logging of training criteria ([9f27c2f](https://github.com/f6ra07nk14/structcast-model/commit/9f27c2f084ba9035c0b07edc1233b3866a4cc864))
* streamline import collection and enhance layer import handling ([e090319](https://github.com/f6ra07nk14/structcast-model/commit/e0903191b16486460ccaa2a5470214aaf6b950ed))
* streamline import statements in base_builder and schema modules ([b42f109](https://github.com/f6ra07nk14/structcast-model/commit/b42f1093e759234c20177a050bb787e27f21a0ec))
* streamline lazy import structure and enhance attribute handling in LazySelectedImporter ([7b0b13e](https://github.com/f6ra07nk14/structcast-model/commit/7b0b13eccef8f95a46501c6bf6862dcffe42e6c9))
* streamline state dict handling in training function for improved clarity ([8330caf](https://github.com/f6ra07nk14/structcast-model/commit/8330cafb87b301af70f8aefe43f481a608c64948))
* streamline usage of to_pascal and to_snake functions in base_builder.py ([5ca8f4b](https://github.com/f6ra07nk14/structcast-model/commit/5ca8f4baf0ec5709715e130b500dbd985c17fd53))
* update _TEMPLATE_ALIASES to use constants from structcast.core.template ([7d5176f](https://github.com/f6ra07nk14/structcast-model/commit/7d5176fccf41ed7d6b95fa5f71af5e5b6fe6edf2))
* update docstring for BestCriterion to clarify its purpose in tracking the best criterion during training or validation ([c0d1780](https://github.com/f6ra07nk14/structcast-model/commit/c0d17804b62a5acf2a532c9622b29e2435696c63))
* update import paths for Tensor and related types in multiple files ([c33c136](https://github.com/f6ra07nk14/structcast-model/commit/c33c136742252f5ce847fdeb902b47403c0e7b9d))
* update LayerBehavior tuple/list validation to allow 2 elements ([ffae6f5](https://github.com/f6ra07nk14/structcast-model/commit/ffae6f5d2d6e6129bb6a384cd2718ce05a050f57))
* update LAYERS field validator to improve validation logic and add legal layer check ([46eef73](https://github.com/f6ra07nk14/structcast-model/commit/46eef73063cc133ac07893a3024db043b180e4b7))
* update model instantiation to use object pattern and add device option ([6d33266](https://github.com/f6ra07nk14/structcast-model/commit/6d33266e236187c9f137d0e2999e0748e4617cee))
* update template handling to use __call__ method for improved clarity ([6a15f09](https://github.com/f6ra07nk14/structcast-model/commit/6a15f0966094beb0ee5bae5b6bea7f4e9c2e1c97))
* update type imports and enhance type checking for optimizers ([45b42e9](https://github.com/f6ra07nk14/structcast-model/commit/45b42e96ad5ed211f2a6f4eee5c2c045b89d327a))


### ✨ Style

* add type ignore comments for lambda functions in optimizer callbacks ([6e42549](https://github.com/f6ra07nk14/structcast-model/commit/6e42549711702340e8845ec03d1fe5e05de3afb8))
* cast Parameters type for better type safety in BaseModelBuilder ([797acb3](https://github.com/f6ra07nk14/structcast-model/commit/797acb3ba22b27ffcb87eb0c266e2226f5ad506d))
* update __dir__ method return type to tuple for better type consistency in LazySelectedImporter ([1a808ba](https://github.com/f6ra07nk14/structcast-model/commit/1a808ba427c9e2f4ea4e494e910e6f5b8e9af346))
* update imports for type hints and compatibility with type checking ([27cec25](https://github.com/f6ra07nk14/structcast-model/commit/27cec253003aeb7a58a08ca379816fa019e114f6))
* update logger type annotation to specify float return type ([a968f25](https://github.com/f6ra07nk14/structcast-model/commit/a968f25799ee38f89c32fcdcdb026474f0212785))
* update type hints for merge method and layers attribute in Parameters and LayerIntermediate classes ([b3e3741](https://github.com/f6ra07nk14/structcast-model/commit/b3e374172d696675daaf8db2827b482692c4c906))


### 🚨 Tests

* add comprehensive unit tests for core builders functionality ([7d405e4](https://github.com/f6ra07nk14/structcast-model/commit/7d405e41ad07f2ce06cad3fef99ad47ffcd169f4))
* add MIXED_PRECISION_TYPE to configuration and test for backward builder ([0a76a5f](https://github.com/f6ra07nk14/structcast-model/commit/0a76a5f72bf7b22775a9514eaaf36272fa9732b2))
* add new tests for layer behavior serialization and input resolution ([15f5530](https://github.com/f6ra07nk14/structcast-model/commit/15f55307a96229355bcb61bc6e8d0c0154a31860))
* add tests for base builder utilities and configuration builders ([3434ee2](https://github.com/f6ra07nk14/structcast-model/commit/3434ee22540c1e6838a73698bdaca837fcda23da))
* add tests for circular reference detection in BaseBuilder and introduce circular.yaml fixture ([4b5446c](https://github.com/f6ra07nk14/structcast-model/commit/4b5446ce4b924907820c8de6b5b4b4bff1a32ae6))
* add unit tests for _resolve_inputs and _resolve_outputs functions, including error handling ([51cc795](https://github.com/f6ra07nk14/structcast-model/commit/51cc795356a6e4dbd1c07e69f9bebb3e22e075b1))
* add unit tests for AutoName and load_any functions ([8f66279](https://github.com/f6ra07nk14/structcast-model/commit/8f66279a0aecdf061092556a291aed6c983edf66))
* add unit tests for commands module and utility functions ([0c8b7c4](https://github.com/f6ra07nk14/structcast-model/commit/0c8b7c41e9dc143cbce28ec48652ad0f881b7d80))
* add unit tests for various layers including accuracy, add, channel shuffle, concatenate, criteria tracker, fold, lazy norm, multiply, permute, reduce, reinmax, scale identity, and split ([42bfd96](https://github.com/f6ra07nk14/structcast-model/commit/42bfd96a2ac3b68de2036407bbca3e183caba91b))
