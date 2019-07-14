for col in acne texture dark_spot sensitive wrinkle redness
do
	export BERT_BASE_DIR=~/Documents/insight/skinsight/bert/bert_dev

	for type in test #train dev 
	do 
		python run_classifier.py \
		--task_name=cola \
		--do_train=false \
		--do_eval=false \
		--do_predict=true \
		--data_dir=./results/$col/data_$type \
		--vocab_file=$BERT_BASE_DIR/vocab.txt \
		--bert_config_file=$BERT_BASE_DIR/bert_config.json \
		--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
		--max_seq_length=128 \
		--train_batch_size=32 \
		--learning_rate=2e-5 \
		--num_train_epochs=3.0 \
		--output_dir=./results/$col/bert_output/


		cp results/$col/bert_output/test_results.tsv results/$col/data_$type/test_results.tsv
	done

done