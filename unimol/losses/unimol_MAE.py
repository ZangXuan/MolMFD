import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("unimol_MAE")
class UniMolMAELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888
        self.shortest_mean = 7.98001753223659
        self.shortest_std = 4.130371174098703

    def forward(self, model, sample, reduce=True):
        input_key = "net_input"
        target_key = "target"
        masked_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)
        all_atomic_tokens = sample[target_key]["tokens_atomic"].ne(self.padding_idx)
        sample_size = masked_tokens.long().sum()
        (
            logits_encoder,
            logits_topo,
            logits_geom,
            encoder_distance,
            encoder_distance_geom,
            encoder_shortest,
            encoder_shortest_topo,
            encoder_coord,
            encoder_degree,
            encoder_x_norm,
            decoder_x_norm_topo,
            decoder_x_norm_geom,
            delta_encoder_pair_rep_norm,
            delta_decoder_pair_rep_norm_topo,
            delta_decoder_pair_rep_norm_geom,
            x_mean,
            x_std,
            noisy_x_mean_topo,
            noisy_x_std_topo,
            noisy_x_mean_geom,
            noisy_x_std_geom,
            encoder_rep_topo,
            encoder_rep_geom,
            decoder_rep_graph_topo,
            decoder_rep_graph_geom,
            encoder_rep_graph,
            decoder_rep_graph_topo_hat, 
            mu_topo, 
            log_var_topo,
            decoder_rep_graph_geom_hat, 
            mu_geom, 
            log_var_geom,
        ) = model(**sample[input_key], encoder_masked_tokens=masked_tokens, all_atomic_tokens=all_atomic_tokens)
        target = sample[target_key]["tokens_target"]
        if masked_tokens is not None:
            target = target[masked_tokens]
        
        logging_output = {
            "sample_size": 1,
            "bsz": sample[target_key]["tokens_target"].size(0),
            "seq_len": sample[target_key]["tokens_target"].size(1)
            * sample[target_key]["tokens_target"].size(0),
        }


        masked_token_loss_topo = F.nll_loss(
            F.log_softmax(logits_topo, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        masked_pred_topo = logits_topo.argmax(dim=-1)
        masked_hit_topo = (masked_pred_topo == target).long().sum()
        masked_cnt_topo = sample_size
        loss = masked_token_loss_topo * self.args.masked_token_loss
        if torch.any(torch.isnan(loss)):
            print("masked_token_loss_topo contains NaNs!")
        logging_output["masked_token_loss_topo"] = masked_token_loss_topo.data
        logging_output["masked_token_hit_topo"] = masked_hit_topo.data
        logging_output["masked_token_cnt_topo"] = masked_cnt_topo


        masked_token_loss_geom = F.nll_loss(
            F.log_softmax(logits_geom, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        masked_pred_geom = logits_geom.argmax(dim=-1)
        masked_hit_geom = (masked_pred_geom == target).long().sum()
        masked_cnt_geom = sample_size
        loss = loss + masked_token_loss_geom * self.args.masked_token_loss
        if torch.any(torch.isnan(loss)):
            print("masked_token_loss_geom contains NaNs!")
        logging_output["masked_token_loss_geom"] = masked_token_loss_geom.data
        logging_output["masked_token_hit_geom"] = masked_hit_geom.data
        logging_output["masked_token_cnt_geom"] = masked_cnt_geom

        valid_tokens = (all_atomic_tokens.float() * (1 - masked_tokens.float())).unsqueeze(-1)
        if self.args.kl_loss > 0:
            epsilon = 1e-7
            kl_topo = 0.5 * (((noisy_x_std_topo.float() ** 2) / x_std.float().clamp(min=epsilon) ** 2) * valid_tokens).sum(dim=1) + \
                    ((((noisy_x_mean_topo.float() - x_mean.float())/ x_std.float().clamp(min=epsilon)) ** 2)* valid_tokens).sum(dim=1)
            kl_topo = kl_topo / valid_tokens.squeeze(-1).sum(dim=1, keepdim=True).clamp(min=epsilon)
            kl_geom = 0.5 * (((noisy_x_std_geom.float() ** 2) / x_std.float().clamp(min=epsilon) ** 2) * valid_tokens).sum(dim=1) + \
                    ((((noisy_x_mean_geom.float() - x_mean.float())/ x_std.float().clamp(min=epsilon)) ** 2)* valid_tokens).sum(dim=1)
            kl_geom = kl_geom / valid_tokens.squeeze(-1).sum(dim=1, keepdim=True).clamp(min=epsilon)
            kl_topo_loss = torch.mean(kl_topo)
            kl_geom_loss = torch.mean(kl_geom)
            loss = loss + (kl_topo_loss + kl_geom_loss) * self.args.kl_loss
            if torch.any(torch.isnan(loss)):
                print("kl_topo_loss + kl_geom_loss contains NaNs!")
            
            logging_output["kl_topo_loss"] = kl_topo_loss.data
            logging_output["kl_geom_loss"] = kl_geom_loss.data
            
        if self.args.orthogonal_loss > 0 :
            epsilon = 1e-7
            encoder_rep_topo = encoder_rep_topo.float() / encoder_rep_topo.float().norm(dim=-1, keepdim=True).clamp(min=epsilon)
            encoder_rep_geom = encoder_rep_geom.float() / encoder_rep_geom.float().norm(dim=-1, keepdim=True).clamp(min=epsilon)
            orthogonal_loss = ((encoder_rep_topo * encoder_rep_geom) * valid_tokens).sum(dim=-1)
            orthogonal_loss = torch.clamp(orthogonal_loss - 0.5, min=0.0).sum(dim=1) / valid_tokens.squeeze(-1).sum(dim=1).clamp(min=epsilon)
            orthogonal_loss = torch.mean(orthogonal_loss)

            loss = loss + orthogonal_loss * self.args.orthogonal_loss
            if torch.any(torch.isnan(loss)):
                print("orthogonal_loss contains NaNs!")
            logging_output["orthogonal_loss"] = orthogonal_loss.data

        
        
        if encoder_coord is not None:
            coord_target = sample[target_key]["coord_target"]
            masked_coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_coord_loss * self.args.masked_coord_loss
            if torch.any(torch.isnan(loss)):
                print("masked_coord_loss contains NaNs!")
            logging_output["masked_coord_loss"] = masked_coord_loss.data

        if encoder_distance is not None:
            dist_masked_tokens = masked_tokens
            masked_dist_loss = self.cal_dist_loss(
                sample, encoder_distance, dist_masked_tokens, target_key, normalize=True
            )
            loss = loss + masked_dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.data

        
        if encoder_distance_geom is not None:
            dist_masked_tokens = masked_tokens
            masked_dist_geom_loss = self.cal_dist_loss(
                sample, encoder_distance_geom, dist_masked_tokens, target_key, normalize=True
            )
            loss = loss + masked_dist_geom_loss * self.args.masked_dist_loss
            logging_output["masked_dist_geom_loss"] = masked_dist_geom_loss.data
            if torch.any(torch.isnan(loss)):
                print("masked_dist_geom_loss contains NaNs!")

        if encoder_shortest is not None:
            shortest_masked_tokens = masked_tokens
            masked_shortest_loss = self.cal_shortest_loss(
                sample, encoder_shortest, shortest_masked_tokens, target_key, normalize=True
            )
            loss = loss + masked_shortest_loss * self.args.masked_shortest_loss
            logging_output["masked_shortest_loss"] = masked_shortest_loss.data

        if encoder_shortest_topo is not None:
            shortest_masked_tokens = masked_tokens
            masked_shortest_topo_loss = self.cal_shortest_loss(
                sample, encoder_shortest_topo, shortest_masked_tokens, target_key, normalize=True
            )
            loss = loss + masked_shortest_topo_loss * self.args.masked_shortest_loss
            logging_output["masked_shortest_topo_loss"] = masked_shortest_topo_loss.data
            if torch.any(torch.isnan(loss)):
                print("masked_shortest_topo_loss contains NaNs!")

        if encoder_degree is not None:
            degree_target = sample[target_key]["degree_target"]
            masked_degree_loss = F.smooth_l1_loss(
                encoder_degree[masked_tokens].view(-1, 1).float(),
                degree_target[masked_tokens].view(-1, 1),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_degree_loss * self.args.masked_degree_loss
            logging_output["masked_degree_loss"] = masked_degree_loss.data
            if torch.any(torch.isnan(loss)):
                print("masked_degree_loss contains NaNs!")


        if self.args.encoder_x_norm_loss > 0 and encoder_x_norm is not None:
            loss = loss + self.args.encoder_x_norm_loss * encoder_x_norm
            logging_output["encoder_x_norm_loss"] = encoder_x_norm.data
            if torch.any(torch.isnan(loss)):
                print("encoder_x_norm_loss contains NaNs!")

        if (
            self.args.encoder_delta_pair_repr_norm_loss > 0
            and delta_encoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.encoder_delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            logging_output[
                "encoder_delta_pair_repr_norm_loss"
            ] = delta_encoder_pair_rep_norm.data
            if torch.any(torch.isnan(loss)):
                print("encoder_delta_pair_repr_norm_loss contains NaNs!")

        
        if self.args.decoder_x_norm_loss > 0 and decoder_x_norm_topo is not None:
            loss = loss + self.args.decoder_x_norm_loss * decoder_x_norm_topo
            logging_output["decoder_x_norm_loss_topo"] = decoder_x_norm_topo.data
            if torch.any(torch.isnan(loss)):
                print("decoder_x_norm_loss_topo contains NaNs!")

        if (
            self.args.decoder_delta_pair_repr_norm_loss > 0
            and delta_decoder_pair_rep_norm_topo is not None
        ):
            loss = (
                loss + self.args.decoder_delta_pair_repr_norm_loss * delta_decoder_pair_rep_norm_topo
            )
            logging_output[
                "decoder_delta_pair_repr_norm_loss_topo"
            ] = delta_decoder_pair_rep_norm_topo.data
            if torch.any(torch.isnan(loss)):
                print("decoder_delta_pair_repr_norm_loss_topo contains NaNs!")

        if self.args.decoder_x_norm_loss > 0 and decoder_x_norm_geom is not None:
            loss = loss + self.args.decoder_x_norm_loss * decoder_x_norm_geom
            logging_output["decoder_x_norm_loss_geom"] = decoder_x_norm_geom.data
            if torch.any(torch.isnan(loss)):
                print("decoder_x_norm_loss_geom contains NaNs!")

        if (
            self.args.decoder_delta_pair_repr_norm_loss > 0
            and delta_decoder_pair_rep_norm_geom is not None
        ):
            loss = (
                loss + self.args.decoder_delta_pair_repr_norm_loss * delta_decoder_pair_rep_norm_geom
            )
            logging_output[
                "decoder_delta_pair_repr_norm_loss_geom"
            ] = delta_decoder_pair_rep_norm_geom.data
            if torch.any(torch.isnan(loss)):
                print("decoder_delta_pair_repr_norm_loss_geom contains NaNs!")

        if self.args.contrastive_loss > 0:
            contrastive_topo_loss = self.cal_cont_loss(decoder_rep_graph_topo, encoder_rep_graph) 
            contrastive_geom_loss = self.cal_cont_loss(decoder_rep_graph_geom, encoder_rep_graph)
            loss = loss + self.args.contrastive_loss * (contrastive_topo_loss+contrastive_geom_loss)
            logging_output["contrastive_topo_loss"] = contrastive_topo_loss.data
            logging_output["contrastive_geom_loss"] = contrastive_geom_loss.data

       
        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

       
        masked_token_loss_topo = sum(log.get("masked_token_loss_topo", 0) for log in logging_outputs)
        metrics.log_scalar(
            "masked_token_loss_topo", masked_token_loss_topo / sample_size, sample_size, round=3
        )
        masked_acc_topo = sum(
            log.get("masked_token_hit_topo", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt_topo", 0) for log in logging_outputs)
        metrics.log_scalar("masked_acc_topo", masked_acc_topo, sample_size, round=3)

        masked_token_loss_geom = sum(log.get("masked_token_loss_geom", 0) for log in logging_outputs)
        metrics.log_scalar(
            "masked_token_loss_geom", masked_token_loss_geom / sample_size, sample_size, round=3
        )
        masked_acc_geom = sum(
            log.get("masked_token_hit_geom", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt_geom", 0) for log in logging_outputs)
        metrics.log_scalar("masked_acc_geom", masked_acc_geom, sample_size, round=3)


        kl_topo_loss = sum(log.get("kl_topo_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "kl_topo_loss", kl_topo_loss / sample_size, sample_size, round=3
        )
        kl_geom_loss = sum(log.get("kl_geom_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "kl_geom_loss", kl_geom_loss / sample_size, sample_size, round=3
        )

        orthogonal_loss = sum(
            log.get("orthogonal_loss", 0) for log in logging_outputs
        )
        if orthogonal_loss > 0:
            metrics.log_scalar("orthogonal_loss", orthogonal_loss / sample_size, sample_size, round=3)


        masked_coord_loss = sum(
            log.get("masked_coord_loss", 0) for log in logging_outputs
        )
        if masked_coord_loss > 0:
            metrics.log_scalar(
                "masked_coord_loss",
                masked_coord_loss / sample_size,
                sample_size,
                round=3,
            )

        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3
            )

        masked_dist_geom_loss = sum(
            log.get("masked_dist_geom_loss", 0) for log in logging_outputs
        )
        if masked_dist_geom_loss > 0:
            metrics.log_scalar(
                "masked_dist_geom_loss", masked_dist_geom_loss / sample_size, sample_size, round=3
            )

        masked_shortest_loss = sum(
            log.get("masked_shortest_loss", 0) for log in logging_outputs
        )
        if masked_shortest_loss > 0:
            metrics.log_scalar(
                "masked_shortest_loss", masked_shortest_loss / sample_size, sample_size, round=3
            )

        masked_shortest_topo_loss = sum(
            log.get("masked_shortest_topo_loss", 0) for log in logging_outputs
        )
        if masked_shortest_topo_loss > 0:
            metrics.log_scalar(
                "masked_shortest_topo_loss", masked_shortest_topo_loss / sample_size, sample_size, round=3
            )

        masked_degree_loss = sum(
            log.get("masked_degree_loss", 0) for log in logging_outputs
        )
        if masked_degree_loss > 0:
            metrics.log_scalar(
                "masked_degree_loss",
                masked_degree_loss / sample_size,
                sample_size,
                round=3,
            )

        encoder_x_norm_loss = sum(log.get("encoder_x_norm_loss", 0) for log in logging_outputs)
        if encoder_x_norm_loss > 0:
            metrics.log_scalar(
                "encoder_x_norm_loss", encoder_x_norm_loss / sample_size, sample_size, round=3
            )

        encoder_delta_pair_repr_norm_loss = sum(
            log.get("encoder_delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if encoder_delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "encoder_delta_pair_repr_norm_loss",
                encoder_delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )

        
        decoder_x_norm_loss_topo = sum(log.get("decoder_x_norm_loss_topo", 0) for log in logging_outputs)
        if decoder_x_norm_loss_topo > 0:
            metrics.log_scalar(
                "decoder_x_norm_loss_topo", decoder_x_norm_loss_topo / sample_size, sample_size, round=3
            )

        decoder_delta_pair_repr_norm_loss_topo = sum(
            log.get("decoder_delta_pair_repr_norm_loss_topo", 0) for log in logging_outputs
        )
        if decoder_delta_pair_repr_norm_loss_topo > 0:
            metrics.log_scalar(
                "decoder_delta_pair_repr_norm_loss_topo",
                decoder_delta_pair_repr_norm_loss_topo / sample_size,
                sample_size,
                round=3,
            )

        decoder_x_norm_loss_geom = sum(log.get("decoder_x_norm_loss_geom", 0) for log in logging_outputs)
        if decoder_x_norm_loss_geom > 0:
            metrics.log_scalar(
                "decoder_x_norm_loss_geom", decoder_x_norm_loss_geom / sample_size, sample_size, round=3
            )

        decoder_delta_pair_repr_norm_loss_geom = sum(
            log.get("decoder_delta_pair_repr_norm_loss_geom", 0) for log in logging_outputs
        )
        if decoder_delta_pair_repr_norm_loss_geom > 0:
            metrics.log_scalar(
                "decoder_delta_pair_repr_norm_loss_geom",
                decoder_delta_pair_repr_norm_loss_geom / sample_size,
                sample_size,
                round=3,
            )

        contrastive_topo_loss = sum(log.get("contrastive_topo_loss", 0) for log in logging_outputs)
        if contrastive_topo_loss > 0:
            metrics.log_scalar(
                "contrastive_topo_loss",
                contrastive_topo_loss / sample_size,
                sample_size,
                round=3,
            )

        contrastive_geom_loss = sum(log.get("contrastive_geom_loss", 0) for log in logging_outputs)
        if contrastive_geom_loss > 0:
            metrics.log_scalar(
                "contrastive_geom_loss",
                contrastive_geom_loss / sample_size,
                sample_size,
                round=3,
            )

        
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def cal_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["distance_target"][
            dist_masked_tokens
        ]
        non_pad_pos = masked_distance_target > 0
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[non_pad_pos].view(-1).float(),
            masked_distance_target[non_pad_pos].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss
    
    def cal_shortest_loss(self, sample, shortest, masked_tokens, target_key, normalize=False):
        shortest_masked_tokens = masked_tokens
        masked_shortest = shortest[shortest_masked_tokens, :]
        masked_shortest_target = sample[target_key]["shortest_target"][
            shortest_masked_tokens
        ]
        non_pad_pos = masked_shortest_target > 0
        if normalize:
            masked_shortest_target = (
                masked_shortest_target.float() - self.shortest_mean
            ) / self.shortest_std
        
        masked_shortest_loss = F.smooth_l1_loss(
            masked_shortest[non_pad_pos].view(-1).float(),
            masked_shortest_target[non_pad_pos].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_shortest_loss
    
    def cal_cont_loss(self, X, Y, temperature=0.5, normalize=True):
        if normalize:
            X = F.normalize(X, dim=-1)
            Y = F.normalize(Y, dim=-1)

        criterion = torch.nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, temperature)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        return CL_loss
    
    