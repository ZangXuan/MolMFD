import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from typing import Dict, Any, List
import numpy as np

from fairseq.modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

logger = logging.getLogger(__name__)

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

@register_model("unimol_MAE_padding")
class UniMolMAEPaddingModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="L", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="A",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--max-hop", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-reg-loss",
            type=float,
            metavar="D",
            help="mask regularization ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-shortest-loss",
            type=float,
            metavar="D",
            help="masked shortest path loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--masked-degree-loss",
            type=float,
            metavar="D",
            help="masked degree loss ratio",
        )
        parser.add_argument(
            "--contrastive-loss",
            type=float,
            metavar="D",
            help="masked shortest path loss ratio",
        )
        parser.add_argument(
            "--encoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--encoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--decoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--decoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--kl-loss",
            type=float,
            metavar="D",
            help="kl loss ratio",
        )
        parser.add_argument(
            "--orthogonal-loss",
            type=float,
            metavar="D",
            help="orthogonal loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--encoder-unmasked-tokens-only", action='store_true', help="only input unmasked tokens into encoder"
        )
        parser.add_argument(
            "--encoder-masked-3d-pe", action='store_true', help="only masked #D PE for encoder"
        )
        parser.add_argument(
            "--encoder-apply-pe", action='store_true', help="apply PE for encoder"
        )
        parser.add_argument(
            "--feed-pair-rep-to-decoder", action='store_true', help="feed the pair representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-no-pe", action='store_true', help="Don't apply PE for decoder"
        )
        parser.add_argument(
            "--feed-token-rep-to-decoder", action='store_true', help="feed the token representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-noise", action='store_true', help="Feed noise or [mask] to decoder"
        )
        parser.add_argument(
            "--random-order", action='store_true', help="Feed noise or [mask] to decoder"
        )

    def __init__(self, args, dictionary):
        super().__init__()
        print('Using modified MAE')
        base_architecture(args)
        self.args = args
        self.encoder_masked_3d_pe = args.encoder_masked_3d_pe
        self.encoder_apply_pe = args.encoder_apply_pe
        self.feed_pair_rep_to_decoder = args.feed_pair_rep_to_decoder
        self.decoder_no_pe = args.decoder_no_pe
        self.feed_token_rep_to_decoder = args.feed_token_rep_to_decoder
        self.decoder_noise = args.decoder_noise
        self.random_order = args.random_order


        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )

        self.embed_dmask_tokens = nn.Embedding(2, args.encoder_embed_dim)

        self.max_hop = args.max_hop
        self.embed_shortest = nn.Embedding(self.max_hop+1, args.encoder_attention_heads, 0)

        self.trans_topo = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim // 2),
            nn.GELU(),
            nn.Linear(args.encoder_embed_dim // 2, 2)
        )
        
        self.trans_geom = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim // 2),
            nn.GELU(),
            nn.Linear(args.encoder_embed_dim // 2, 2)
        )

        self.PE = None
        self.index = None
        if self.random_order:
            self.init_state()

        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            # no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
            no_final_head_layer_norm=True,
        )
        # self.decoder = TransformerEncoderWithPair(
        #     encoder_layers=args.decoder_layers,
        #     embed_dim=args.encoder_embed_dim,
        #     ffn_embed_dim=args.decoder_ffn_embed_dim,
        #     attention_heads=args.decoder_attention_heads,
        #     emb_dropout=args.emb_dropout,
        #     dropout=args.dropout,
        #     attention_dropout=args.attention_dropout,
        #     activation_dropout=args.activation_dropout,
        #     max_seq_len=args.max_seq_len,
        #     activation_fn=args.activation_fn,
        #     # no_final_head_layer_norm=args.decoder_delta_pair_repr_norm_loss < 0,
        #     no_final_head_layer_norm=True,
        # )

        self.decoder_topo = TransformerEncoderWithPair(
            encoder_layers=args.decoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            # no_final_head_layer_norm=args.masked_coord_loss < 0,
            no_final_head_layer_norm=args.masked_degree_loss < 0,
        )


        self.decoder_geom = TransformerEncoderWithPair(
            encoder_layers=args.decoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.masked_coord_loss < 0,
            #no_final_head_layer_norm=True,
        )

        if args.masked_token_loss > 0:
            # self.lm_head = MaskLMHead(
            #     embed_dim=args.encoder_embed_dim,
            #     output_dim=len(dictionary),
            #     activation_fn=args.activation_fn,
            #     weight=None,
            # )

            self.lm_topo_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )

            self.lm_geom_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )


        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.decoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            # self.dist_head = DistanceHead(
            #     args.decoder_attention_heads, args.activation_fn
            # )
            self.dist_geom_head = DistanceHead(
                args.decoder_attention_heads, args.activation_fn
            )
        if args.masked_shortest_loss > 0:
            # self.shortest_head = ShortestHead(args.decoder_attention_heads, self.max_hop+1, args.activation_fn
            # )
            # self.shortest_topo_head = ShortestHead(args.decoder_attention_heads, self.max_hop+1, args.activation_fn
            # )
            # self.shortest_topo_head = ShortestHead(args.decoder_attention_heads, 1, args.activation_fn
            # )
            self.shortest_topo_head = ShortestHead(args.decoder_attention_heads, args.activation_fn
            )
        if args.masked_degree_loss > 0:
            self.pair2degree_proj = NonLinearHead(
                args.decoder_attention_heads, 1, args.activation_fn
            )

        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)
        self.encoder_unmasked_tokens_only = args.encoder_unmasked_tokens_only
        self.dictionary = dictionary

        self.embed_positions = SinusoidalPositionalEmbedding(
            embedding_dim = args.max_seq_len,
            padding_idx = dictionary.pad(),
            init_size = args.max_seq_len,
        )

        
        self.mask_idx = dictionary.index("[MASK]")
        self.encoder_attention_heads = args.encoder_attention_heads

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)
    
    @classmethod
    def init_state(self):
        original_state = np.random.get_state()
        np.random.seed(0)
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(original_state)
    
    @classmethod
    def Myenter(self):
        self.original_state = np.random.get_state()
        np.random.set_state(self.numpy_random_state)

    @classmethod
    def Myexit(self):
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(self.original_state)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_shortest_path,
        encoder_masked_tokens=None,
        all_atomic_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        # print('encoder_masked_tokens:',encoder_masked_tokens)
        # exit()

        if self.random_order and self.index is None:
            self.Myenter()
            self.index = np.random.permutation(512)
            self.Myexit()

        if classification_head_name is not None:
            features_only = True
        
        if self.encoder_unmasked_tokens_only and encoder_masked_tokens is not None:
            assert not self.encoder_masked_3d_pe

            # encoder_src_tokens = src_tokens.masked_fill(encoder_masked_tokens, self.padding_idx)
            encoder_src_tokens = src_tokens
            encoder_src_coord = src_coord.masked_fill(encoder_masked_tokens.unsqueeze(-1).expand_as(src_coord), 0)

            encoder_valid_src_tokens_mask_seq = (~encoder_masked_tokens.long().unsqueeze(-1) * ~encoder_masked_tokens.long().unsqueeze(1)).bool()
            encoder_src_distance = src_distance.masked_fill(~encoder_valid_src_tokens_mask_seq, 0)
            encoder_src_edge_type = src_edge_type.masked_fill(~encoder_valid_src_tokens_mask_seq, 0)
            encoder_src_shortest_path = src_shortest_path.masked_fill(~encoder_valid_src_tokens_mask_seq, 0)
        else:
            encoder_src_tokens = src_tokens
            encoder_src_coord = src_coord
            encoder_src_distance = src_distance
            encoder_src_edge_type = src_edge_type
            encoder_src_shortest_path = src_shortest_path

        encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(encoder_src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        
        def get_shortest_features(shortest):
            n_node = shortest.size(-1)
            shortest_result = self.embed_shortest(shortest)
            graph_attn_bias = shortest_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type) + \
                          get_shortest_features(encoder_src_shortest_path)

        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            encoder_x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=encoder_padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0


        #### inject noise
        def gumbel_sample(prob, tau=1.0, hard=False):
            assignment = torch.nn.functional.softmax(prob, dim=-1)
            gumbel_assignment = F.gumbel_softmax(assignment, tau=tau, dim = -1, hard=hard)
            return gumbel_assignment
        
        static_x = encoder_rep.clone().detach().to(torch.float32)
        valid_tokens = (all_atomic_tokens.to(torch.float32) * (1 - encoder_masked_tokens.to(torch.float32))).unsqueeze(-1)
        x_mean = (static_x * valid_tokens).sum(dim=1, keepdim=True) / (valid_tokens.sum(dim=1, keepdim=True) + 1e-6)  ###[bs, 1 ,d]
        x_std = torch.sqrt((((static_x  - x_mean) * valid_tokens) ** 2).sum(dim=1, keepdim=True) / (valid_tokens.sum(dim=1, keepdim=True) + 1e-6))
        # print("static_x:",static_x.shape,static_x)
        # print("valid_tokens:",valid_tokens.shape,valid_tokens)
        # print("x_mean:",x_mean.shape,x_mean)
        # print("x_std:",x_std.shape,x_std)
        # assert 1==2

        prob_topo = self.trans_topo(encoder_rep)
        #gate_inputs_neg = gumbel_sample(prob_topo, tau=1.0)[:, :, 1].unsqueeze(dim = -1)
        gate_inputs_neg = torch.nn.functional.softmax(prob_topo, dim=-1)[:, :, 1].unsqueeze(dim = -1)
        gate_inputs_neg = gate_inputs_neg * valid_tokens
        gate_inputs_pos = 1 - gate_inputs_neg
        noisy_x_mean_topo = gate_inputs_pos * encoder_rep + gate_inputs_neg * x_mean
        noisy_x_std_topo = gate_inputs_neg * x_std
        # encoder_rep_topo = (encoder_rep + torch.rand_like(noisy_x_mean_topo) * noisy_x_std_topo).type_as(encoder_rep) ###inject noise
        encoder_rep_topo = (noisy_x_mean_topo + torch.randn_like(noisy_x_mean_topo) * noisy_x_std_topo).type_as(encoder_rep) ###inject noise

        #print("gate_inputs_neg1",gate_inputs_neg)
        prob_geom = self.trans_geom(encoder_rep)
        #gate_inputs_neg = gumbel_sample(prob_geom, tau=1.0)[:, :, 1].unsqueeze(dim = -1)
        gate_inputs_neg = torch.nn.functional.softmax(prob_geom, dim=-1)[:, :, 1].unsqueeze(dim = -1)
        gate_inputs_neg = gate_inputs_neg * valid_tokens
        gate_inputs_pos = 1 - gate_inputs_neg
        noisy_x_mean_geom = gate_inputs_pos * encoder_rep + gate_inputs_neg * x_mean
        noisy_x_std_geom = gate_inputs_neg * x_std
        # encoder_rep_geom = (encoder_rep + torch.rand_like(noisy_x_mean_geom) * noisy_x_std_geom).type_as(encoder_rep) ###inject noise
        encoder_rep_geom = (noisy_x_mean_geom + torch.randn_like(noisy_x_mean_geom) * noisy_x_std_geom).type_as(encoder_rep) ###inject noise

        # print("gate_inputs_neg2",gate_inputs_neg)
        # assert 1==2

        if self.feed_token_rep_to_decoder:
            encoder_output_embedding = encoder_rep
            encoder_output_embedding_topo = encoder_rep_topo
            encoder_output_embedding_geom = encoder_rep_geom
        else:
            if encoder_masked_tokens is None:
                encoder_output_embedding = encoder_rep
                encoder_output_embedding_topo = encoder_rep_topo
                encoder_output_embedding_geom = encoder_rep_geom
            else:
                mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
                masked_embeddings = self.embed_tokens(mask_tokens)
                encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)

                mask_tokens_topo = torch.zeros_like(src_tokens).fill_(0)
                masked_embeddings_topo = self.embed_dmask_tokens(mask_tokens_topo)
                encoder_output_embedding_topo = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings_topo, encoder_rep_topo)

                mask_tokens_geom = torch.zeros_like(src_tokens).fill_(1)
                masked_embeddings_geom = self.embed_dmask_tokens(mask_tokens_geom)
                encoder_output_embedding_geom = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings_geom, encoder_rep_geom)

        ####### decoder start
        if not self.decoder_no_pe:
            encoder_output_embedding = encoder_output_embedding + self.embed_positions(src_tokens)
            encoder_output_embedding_topo = encoder_output_embedding_topo + self.embed_positions(src_tokens)
            encoder_output_embedding_geom = encoder_output_embedding_geom + self.embed_positions(src_tokens)

        n_node = encoder_output_embedding.size(1)
        if self.feed_pair_rep_to_decoder:
            assert self.decoder_noise is not True
            attn_bias = encoder_pair_rep.reshape(-1, n_node, n_node)
        else:
            if not self.decoder_noise:
                bsz = src_tokens.size(0)
                attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
            else:
                attn_bias = get_dist_features(src_distance, src_edge_type)
        # (
        #     decoder_rep,
        #     decoder_pair_rep,
        #     delta_decoder_pair_rep,
        #     decoder_x_norm,
        #     delta_decoder_pair_rep_norm,
        # ) = self.decoder(encoder_output_embedding, padding_mask=padding_mask, attn_mask=attn_bias)
        # decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0

        ## encoder_rep_topo = ((encoder_masked_tokens.unsqueeze(-1) * self.embed_tokens(torch.full_like(src_tokens, self.dmask_topo_idx)))  \
        ##   + ~encoder_masked_tokens.unsqueeze(-1) * encoder_rep_topo).type_as(encoder_rep)  ### remask     

        (
            decoder_rep_topo,
            decoder_pair_rep_topo,
            delta_decoder_pair_rep_topo,
            decoder_x_norm_topo,
            delta_decoder_pair_rep_norm_topo,
        ) = self.decoder_topo(encoder_output_embedding_topo, padding_mask=padding_mask, attn_mask=attn_bias)
        # ) = self.decoder_topo(encoder_rep_topo, padding_mask=padding_mask, attn_mask=None)
        decoder_pair_rep_topo[decoder_pair_rep_topo == float("-inf")] = 0

        (
            decoder_rep_geom,
            decoder_pair_rep_geom,
            delta_decoder_pair_rep_geom,
            decoder_x_norm_geom,
            delta_decoder_pair_rep_norm_geom,
        ) = self.decoder_geom(encoder_output_embedding_geom, padding_mask=padding_mask, attn_mask=attn_bias)
        # ) = self.decoder_geom(encoder_rep_geom.type_as(encoder_rep), padding_mask=padding_mask, attn_mask=None)
        decoder_pair_rep_geom[decoder_pair_rep_geom == float("-inf")] = 0
        
        # decoder_rep_graph = decoder_rep[:, 0, :]
        encoder_rep_graph = encoder_rep[:, 0, :]
        decoder_rep_graph_topo = decoder_rep_topo[:, 0, :]
        decoder_rep_graph_geom = decoder_rep_geom[:, 0, :]

        encoder_distance = None
        encoder_shortest=None
        encoder_coord = None
        encoder_degree=None
        logits=None
        decoder_rep_graph_topo_hat, mu_topo, log_var_topo,decoder_rep_graph_geom_hat, mu_geom, log_var_geom=None,None,None,None,None,None

        if not features_only:
            if self.args.masked_token_loss > 0:
                # logits = self.lm_head(decoder_rep, encoder_masked_tokens)
                #logits = self.lm_head(encoder_rep, encoder_masked_tokens)
                logits_topo = self.lm_topo_head(decoder_rep_topo, encoder_masked_tokens)
                logits_geom = self.lm_geom_head(decoder_rep_geom, encoder_masked_tokens)
            if self.args.masked_coord_loss > 0:
                coords_emb = src_coord
                if padding_mask is not None:
                    atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(
                        -1, 1, 1, 1
                    )
                else:
                    atom_num = src_coord.shape[1] - 1
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pair2coord_proj(delta_decoder_pair_rep_geom)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = torch.sum(coord_update, dim=2)
                encoder_coord = coords_emb + coord_update
            if self.args.masked_dist_loss > 0:
                #encoder_distance = self.dist_head(decoder_pair_rep)
                encoder_distance_geom = self.dist_geom_head(decoder_pair_rep_geom)
            if self.args.masked_shortest_loss > 0:
                #encoder_shortest = self.shortest_head(decoder_pair_rep)
                encoder_shortest_topo = self.shortest_topo_head(decoder_pair_rep_topo)
            if self.args.masked_degree_loss > 0:
                atom_num = (all_atomic_tokens.sum(-1)).view(-1, 1, 1, 1)
                attn_probs = self.pair2degree_proj(delta_decoder_pair_rep_topo * all_atomic_tokens.type_as(delta_decoder_pair_rep_topo).unsqueeze(1).unsqueeze(-1))
                encoder_degree = attn_probs / atom_num 
                encoder_degree = torch.sum(encoder_degree, dim=2)

            

        if classification_head_name is not None:
            logits = self.classification_heads[classification_head_name](encoder_rep)
        if self.args.mode == 'infer':
            return encoder_rep, encoder_pair_rep, decoder_pair_rep_topo, decoder_pair_rep_geom
        else:
            return (
                logits,
                logits_topo,
                logits_geom,
                encoder_distance,
                encoder_distance_geom,
                encoder_shortest,
                encoder_shortest_topo,
                encoder_coord,
                encoder_degree,
                encoder_x_norm,
                #decoder_x_norm,
                decoder_x_norm_topo,
                decoder_x_norm_geom,
                delta_encoder_pair_rep_norm,
                #delta_decoder_pair_rep_norm,
                delta_decoder_pair_rep_norm_topo,
                delta_decoder_pair_rep_norm_geom,
                x_mean,
                x_std,
                noisy_x_mean_topo,
                noisy_x_std_topo,
                noisy_x_mean_geom,
                noisy_x_std_geom,
                # encoder_rep,
                encoder_rep_topo,
                encoder_rep_geom,
                #decoder_rep_graph,
                decoder_rep_graph_topo,
                decoder_rep_graph_geom,
                encoder_rep_graph,
                decoder_rep_graph_topo_hat, 
                mu_topo, 
                log_var_topo,
                decoder_rep_graph_geom_hat, 
                mu_geom, 
                log_var_geom,
            )            

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):

    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x
    
class ShortestHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        heads,
        #num_classes,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        # self.out_proj = nn.Linear(heads, num_classes)
        self.out_proj = nn.Linear(heads, 1)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # x = self.out_proj(x).view(bsz, seq_len, seq_len, -1)
        # x = (x + x.transpose(-2, -3)) * 0.5
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)
    

class VariationalAutoEncoder(torch.nn.Module):
    # def __init__(self, emb_dim, loss="l2", detach_target=True, beta=1):
    def __init__(self, emb_dim, activation_fn):
        super(VariationalAutoEncoder, self).__init__()
        self.emb_dim = emb_dim
        #self.loss = loss
        #self.detach_target = detach_target
        #self.beta = beta

        # self.criterion = None
        # if loss == 'l1':
        #     self.criterion = nn.L1Loss()
        # elif loss == 'l2':
        #     self.criterion = nn.MSELoss()
        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.fc_mu = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.emb_dim)
        self.dense = nn.Linear(self.emb_dim, self.emb_dim)
        self.out = nn.Linear(self.emb_dim, self.emb_dim)

        # self.decoder = nn.Sequential(
        #     nn.Linear(self.emb_dim, self.emb_dim),
        #     #nn.BatchNorm1d(self.emb_dim),
        #     self.activation_fn,
        #     nn.Linear(self.emb_dim, self.emb_dim),
        # )
        # return

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # def forward(self, x, y):
        # if self.detach_target:
        #     y = y.detach()
    def forward(self, x):

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        # y_hat = self.decoder(z)
        y_hat = self.dense(z)
        y_hat = self.activation_fn(y_hat)
        y_hat = self.out(y_hat)

        # reconstruction_loss = self.criterion(y_hat, y)
        # kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # loss = reconstruction_loss + self.beta * kl_loss

        # return loss
        return y_hat, mu, log_var



@register_model_architecture("unimol_MAE_padding", "unimol_MAE_padding")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    args.masked_shortest_loss = getattr(args, "masked_shortest_loss", -1.0)
    args.kl_loss = getattr(args, "kl_loss", -1.0)
    args.orthogonal_loss = getattr(args, "orthogonal_loss", -1.0)
    args.masked_reg_loss = getattr(args, "masked_reg_loss", -1.0)
    args.contrastive_loss = getattr(args, "contrastive_loss", -1.0)
    args.masked_degree_loss = getattr(args, "masked_degree_loss", -1.0)
    args.max_hop = getattr(args, "max_hop", 32)

    args.encoder_masked_3d_pe = getattr(args, "encoder_masked_3d_pe", False)
    args.encoder_unmasked_tokens_only = getattr(args, "encoder_unmasked_tokens_only", False)
    args.encoder_apply_pe = getattr(args, "encoder_apply_pe", False)
    args.feed_pair_rep_to_decoder = getattr(args, "feed_pair_rep_to_decoder", False)
    args.decoder_no_pe = getattr(args, "decoder_no_pe", False)
    args.feed_token_rep_to_decoder = getattr(args, "feed_token_rep_to_decoder", False)
    args.decoder_noise = getattr(args, "decoder_noise", False)
    args.random_order = getattr(args, "random_order", False)
    
    

@register_model_architecture("unimol_MAE_padding", "unimol_MAE_padding_base")
def unimol_base_architecture(args):
    base_architecture(args)

@register_model_architecture("unimol_MAE_padding", "unimol_MAE_padding_150M")
def base_150M_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 30)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 640)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2560)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 20)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

    args.decoder_noise = getattr(args, "decoder_noise", False)
    args.random_order = getattr(args, "random_order", False)
