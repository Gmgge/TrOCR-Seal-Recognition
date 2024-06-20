import os
import argparse
from tool.dataset import trocrDataset, decode_text
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import default_data_collator
from sklearn.model_selection import train_test_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric
from tool.file_tool import get_image_file_list
from tool.image_aug import aug_sequential


def compute_metrics(pred):
    """
    计算cer,acc
    :param pred:
    :return:
    """
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = [decode_text(pred_id, vocab, vocab_inp) for pred_id in pred_ids]
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = [decode_text(labels_id, vocab, vocab_inp) for labels_id in labels_ids]
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    acc = [pred == label for pred, label in zip(pred_str, label_str)]
    print([pred_str[0], label_str[0]])
    acc = sum(acc)/(len(acc)+0.000001)

    return {"cer": cer, "acc": acc}


if __name__ == '__main__':
    # 配置参数
    parser = argparse.ArgumentParser(description='trocr fine-tune训练')
    parser.add_argument('--cust_data_init_weights_path', default='./cust-data/weights', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--checkpoint_path', default='./checkpoint/trocr', type=str, help="训练模型保存地址")
    parser.add_argument('--dataset_path', default='./dataset/cust-data/', type=str, help="训练数据集")
    parser.add_argument('--per_device_train_batch_size', default=32, type=int, help="train batch size")
    parser.add_argument('--per_device_eval_batch_size', default=8, type=int, help="eval batch size")
    parser.add_argument('--max_target_length', default=128, type=int, help="训练文字字符数")
    parser.add_argument('--num_train_epochs', default=10, type=int, help="训练epoch数")
    parser.add_argument('--eval_steps', default=1000, type=int, help="模型评估间隔数")
    parser.add_argument('--save_steps', default=1000, type=int, help="模型保存间隔步数")
    parser.add_argument('--learning_rate', default=4e-4, type=float, help="训练学习率")
    parser.add_argument('--logging_steps', default=10, type=int, help="日志打印步长")
    parser.add_argument('--save_total_limit', default=10, type=int, help="模型记录数量上限")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1', type=str, help="GPU设置")
    args = parser.parse_args()
    print("train param")
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    # 配置数据集
    print("loading data .................")
    paths = get_image_file_list(args.dataset_path)
    print("data count: {}".format(len(paths)))
    train_paths, test_paths = train_test_split(paths, test_size=0.05, random_state=10086)
    print("train num:", len(train_paths), "test num:", len(test_paths))

    # 图像预处理
    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}

    # 构建数据装载
    train_transformer = aug_sequential  # 训练集数据增强
    train_dataset = trocrDataset(paths=train_paths, processor=processor, max_target_length=args.max_target_length,
                                 transformer=train_transformer)
    eval_dataset = trocrDataset(paths=test_paths, processor=processor, max_target_length=args.max_target_length)

    # 配置模型参数
    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = args.max_target_lengt
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    cer_metric = load_metric("./tool/cer.py")

    # 配置训练参数
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=True,
        output_dir=args.checkpoint_path,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate
    )

    # 配置训练
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_path, 'last'))
    processor.save_pretrained(os.path.join(args.checkpoint_path, 'last'))





