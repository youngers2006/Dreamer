Dreamer V3 paper by Danijar Hafner: https://arxiv.org/pdf/2301.04104

Project Description: This project aims to create a custom implementation of Dreamer V3 from the paper linked above. This implementation is done in pytorch to differentiate itself from the official JAX repository of Dreamer V3.

Current State: In its current state this model is tested and functional on box2d environments such as CarRacer when learning from pixels, I am still working on benchmarking the model against other RL algorithms as the architecture is not yet in its final state.

Next Steps: My aim for this project is to successfully fly a drone from pixels in simulation, this capability is in testing but due to limited access to sufficient compute progress on this is slow. To achieve this I must ensure my architecture is as optimised as possible, a point where the code in its current state is approaching.

To run the CarRacer training loop use: python train_car_racer.py --config car_racer_config.yaml
