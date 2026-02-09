import os
# HF_HOME should be set via environment variable or in your shell config
# os.environ['HF_HOME'] = 'your_hf_cache_path'  # TODO: Set your HuggingFace cache path
from ..evaluate import evaluate_clutrr
from ..evaluate.gen_eval import (
    evaluate_boolq, evaluate_arc
)
from ..mlps.stt_linear import STTLinear
from trl import SFTTrainer
import wandb


class CustomSFTTrainerV2(SFTTrainer):
    def __init__(self, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
        self.log_interval = self.args.logging_steps
        self.epoch_accuracies = []
        self.task = task

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        cn_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = cn_loss
        if self.args.logging_steps and self.state.global_step % self.args.logging_steps == 0:
            if wandb.run is not None:
                wandb.log({"train/loss": loss.item()}, step=self.state.global_step)
        self.step += 1
        if return_outputs:
            return loss, outputs
        else:
            return loss

    def log_metrics(self, split, metrics, step=None):
        """Log metrics to wandb with proper prefixing"""
        if wandb.run is not None:
            wandb_metrics = {f"{split}/{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=step if step is not None else self.step)
        super().log_metrics(split, metrics)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Overriding evaluate to compute and log accuracy every epoch"""
        # Call parent evaluate method first
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        print("[DEBUG] eval_results keys:", list(eval_results.keys()))
        if "loss" in eval_results:
            eval_loss = eval_results["loss"]
            eval_results["eval_loss"] = eval_loss
            print(f"[Eval Loss] {eval_loss:.4f}")
            if wandb.run is not None:
                wandb.log({"eval/loss": eval_loss}, step=self.state.global_step)

        # Get the current epoch
        current_epoch = self.state.epoch if self.state.epoch is not None else 0

        # Run the full evaluation for accuracy
        if eval_dataset is not None:
            print(f"Running evaluation for epoch {current_epoch}")
            eval_result = self.test(
                fname=os.path.join(self.args.output_dir, f"epoch_{int(current_epoch)}"),
                task=self.task,
                eval_dataset=eval_dataset,
                model_name=self.args.model_name,
                apply_chat_template=self.args.apply_chat_template
            )

            # Extract accuracy
            if isinstance(eval_result, tuple) and len(eval_result) == 2:
                accuracy, _ = eval_result
            elif isinstance(eval_result, dict) and 'accuracy' in eval_result:
                accuracy = eval_result['accuracy']
            else:
                accuracy = None

            if accuracy is not None:
                self.epoch_accuracies.append((current_epoch, accuracy))
                eval_results['eval_accuracy'] = accuracy

                if wandb.run is not None:
                    wandb.log({
                        "eval/accuracy": accuracy,
                        "epoch": current_epoch
                    }, step=self.state.global_step)

        return eval_results

    def test(self, fname, task, eval_dataset=None, ignore_keys=None, sanity_check=False, model_name=None,
             apply_chat_template=False, **kwargs):
        if sanity_check:
            print('Sanity check...')
        self.model.eval()
        self.args.prediction_loss_only = False

        os.makedirs(fname, exist_ok=True)

        if wandb.run is not None:
            predictions_table = wandb.Table(columns=["example_id", "input", "prediction", "ground_truth", "correct"])

        if task == 'clutrr':
            generated = evaluate_clutrr(eval_dataset=eval_dataset, model_name=model_name, model=self.model,
                                        tokenizer=self.processing_class, fname=fname,
                                        apply_chat_template=apply_chat_template)
        elif task == 'boolq':
            generated = evaluate_boolq(eval_dataset=eval_dataset, model_name=model_name, model=self.model,
                                       tokenizer=self.processing_class, fname=fname,
                                       apply_chat_template=apply_chat_template)
        elif task == 'arc-e':
            generated = evaluate_arc(eval_dataset=eval_dataset, model_name=model_name, model=self.model,
                                     tokenizer=self.processing_class, fname=fname, subset="easy",
                                     apply_chat_template=apply_chat_template)
        elif task == 'arc-c':
            generated = evaluate_arc(eval_dataset=eval_dataset, model_name=model_name, model=self.model,
                                     tokenizer=self.processing_class, fname=fname, subset="challenge",
                                     apply_chat_template=apply_chat_template)
        else:
            raise ValueError(f"Unknown task: {task}")

        if isinstance(generated, dict) and 'metrics' in generated:
            metrics = generated['metrics']
            self.log_metrics("evaluate", metrics)

            print(f"\n[Evaluation Metrics]")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

            # Also log individual example predictions if available
            if wandb.run is not None and 'accuracy' in metrics:
                wandb.log({"eval/accuracy": metrics['accuracy']}, step=self.state.global_step)

            if wandb.run is not None and 'predictions' in generated:
                for i, pred in enumerate(generated['predictions']):
                    if isinstance(pred, dict):
                        predictions_table.add_data(
                            i,
                            pred.get('input', 'N/A'),
                            pred.get('prediction', 'N/A'),
                            pred.get('ground_truth', 'N/A'),
                            pred.get('correct', False)
                        )
                wandb.log({"evaluate/predictions": predictions_table})

                if 'active_neurons' in generated:
                    wandb.log({"evaluate/active_neurons": generated['active_neurons']})

        return generated

    def log_model_info(self):
        """Log model information to WandB"""
        if wandb.run is None:
            return

        try:
            # Count total and trainable parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            # Count neurons per layer for STT MLPs
            neuron_counts = {}
            llama_model = self.model.model if hasattr(self.model, "model") else self.model

            for i, layer in enumerate(llama_model.layers):
                if hasattr(layer, 'mlp') and isinstance(layer.mlp, STTLinear):
                    neuron_counts[f"layer_{i}"] = {
                        "active_neurons": layer.mlp.intermediate_size,
                        "total_neurons": layer.mlp.original_intermediate_size,
                        "percentage": layer.mlp.intermediate_size / layer.mlp.original_intermediate_size
                    }

            wandb.log({
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/trainable_percentage": trainable_params / total_params if total_params > 0 else 0,
                "model/neuron_counts": wandb.Table(
                    columns=["layer", "active_neurons", "total_neurons", "percentage"],
                    data=[[k, v["active_neurons"], v["total_neurons"], v["percentage"]]
                          for k, v in neuron_counts.items()]
                )
            })

            # Log histogram of neuron usage
            if neuron_counts:
                percentages = [v["percentage"] for v in neuron_counts.values()]
                wandb.log({"model/neuron_usage_histogram": wandb.Histogram(percentages)})

        except Exception as e: