python3 tensorflow/tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=nn_files/bottlenecks \
--how_many_training_steps 500 \
--model_dir=nn_files/inception \
--output_graph=nn_files/retrained_graph.pb \
--output_labels=nn_files/retrained_labels.txt \
--image_dir nn_files/photos
