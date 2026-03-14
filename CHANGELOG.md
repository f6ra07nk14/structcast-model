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
