# PyTorch ONNX Conversion Report

```
✅ Obtain model graph with `torch.export.export(..., strict=False)`
⚪ Obtain model graph with `torch.export.export(..., strict=True)`
⚪ Obtain model graph with `torch.jit.trace`
✅ Decompose operators for ONNX compatibility
✅ Translate the graph into ONNX
⚪ Run `onnx.checker` on the ONNX model
⚪ Execute the model with ONNX Runtime
⚪ Validate model output accuracy
```

## Error messages

```pytb
No errors
```

## Exported program

```python
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, p_core_cls: "f32[1, 128]", p_core_ring: "f32[1, 128]", p_core_end: "f32[1, 128]", p_core_node_processor_x_emb_weight: "f32[120, 128]", p_core_node_processor_lin_weight: "f32[128, 128]", p_core_node_processor_lin_bias: "f32[128]", p_core_node_processor_norm_weight: "f32[128]", p_core_node_processor_norm_bias: "f32[128]", p_core_node_processor_graph_convs_0_att: "f32[1, 1, 128]", p_core_node_processor_graph_convs_0_bias: "f32[128]", p_core_node_processor_graph_convs_0_lin_l_weight: "f32[128, 128]", p_core_node_processor_graph_convs_0_lin_l_bias: "f32[128]", p_core_node_processor_graph_convs_0_lin_r_weight: "f32[128, 128]", p_core_node_processor_graph_convs_0_lin_r_bias: "f32[128]", p_core_node_processor_graph_convs_0_lin_edge_weight: "f32[128, 128]", p_core_node_processor_graph_convs_1_att: "f32[1, 1, 128]", p_core_node_processor_graph_convs_1_bias: "f32[128]", p_core_node_processor_graph_convs_1_lin_l_weight: "f32[128, 128]", p_core_node_processor_graph_convs_1_lin_l_bias: "f32[128]", p_core_node_processor_graph_convs_1_lin_r_weight: "f32[128, 128]", p_core_node_processor_graph_convs_1_lin_r_bias: "f32[128]", p_core_node_processor_graph_convs_1_lin_edge_weight: "f32[128, 128]", p_core_node_processor_graph_convs_2_att: "f32[1, 1, 128]", p_core_node_processor_graph_convs_2_bias: "f32[128]", p_core_node_processor_graph_convs_2_lin_l_weight: "f32[128, 128]", p_core_node_processor_graph_convs_2_lin_l_bias: "f32[128]", p_core_node_processor_graph_convs_2_lin_r_weight: "f32[128, 128]", p_core_node_processor_graph_convs_2_lin_r_bias: "f32[128]", p_core_node_processor_graph_convs_2_lin_edge_weight: "f32[128, 128]", p_core_node_processor_graph_convs_3_att: "f32[1, 1, 128]", p_core_node_processor_graph_convs_3_bias: "f32[128]", p_core_node_processor_graph_convs_3_lin_l_weight: "f32[128, 128]", p_core_node_processor_graph_convs_3_lin_l_bias: "f32[128]", p_core_node_processor_graph_convs_3_lin_r_weight: "f32[128, 128]", p_core_node_processor_graph_convs_3_lin_r_bias: "f32[128]", p_core_node_processor_graph_convs_3_lin_edge_weight: "f32[128, 128]", p_core_node_processor_graph_convs_4_att: "f32[1, 1, 128]", p_core_node_processor_graph_convs_4_bias: "f32[128]", p_core_node_processor_graph_convs_4_lin_l_weight: "f32[128, 128]", p_core_node_processor_graph_convs_4_lin_l_bias: "f32[128]", p_core_node_processor_graph_convs_4_lin_r_weight: "f32[128, 128]", p_core_node_processor_graph_convs_4_lin_r_bias: "f32[128]", p_core_node_processor_graph_convs_4_lin_edge_weight: "f32[128, 128]", p_core_node_processor_graph_convs_5_att: "f32[1, 1, 128]", p_core_node_processor_graph_convs_5_bias: "f32[128]", p_core_node_processor_graph_convs_5_lin_l_weight: "f32[128, 128]", p_core_node_processor_graph_convs_5_lin_l_bias: "f32[128]", p_core_node_processor_graph_convs_5_lin_r_weight: "f32[128, 128]", p_core_node_processor_graph_convs_5_lin_r_bias: "f32[128]", p_core_node_processor_graph_convs_5_lin_edge_weight: "f32[128, 128]", p_core_node_processor_graph_norms_0_weight: "f32[128]", p_core_node_processor_graph_norms_0_bias: "f32[128]", p_core_node_processor_graph_norms_1_weight: "f32[128]", p_core_node_processor_graph_norms_1_bias: "f32[128]", p_core_node_processor_graph_norms_2_weight: "f32[128]", p_core_node_processor_graph_norms_2_bias: "f32[128]", p_core_node_processor_graph_norms_3_weight: "f32[128]", p_core_node_processor_graph_norms_3_bias: "f32[128]", p_core_node_processor_graph_norms_4_weight: "f32[128]", p_core_node_processor_graph_norms_4_bias: "f32[128]", p_core_ring_encoder_layers_0_self_attn_in_proj_weight: "f32[384, 128]", p_core_ring_encoder_layers_0_self_attn_in_proj_bias: "f32[384]", p_core_ring_encoder_layers_0_self_attn_out_proj_weight: "f32[128, 128]", p_core_ring_encoder_layers_0_self_attn_out_proj_bias: "f32[128]", p_core_ring_encoder_layers_0_linear1_weight: "f32[1024, 128]", p_core_ring_encoder_layers_0_linear1_bias: "f32[1024]", p_core_ring_encoder_layers_0_linear2_weight: "f32[128, 1024]", p_core_ring_encoder_layers_0_linear2_bias: "f32[128]", p_core_ring_encoder_layers_0_norm1_weight: "f32[128]", p_core_ring_encoder_layers_0_norm1_bias: "f32[128]", p_core_ring_encoder_layers_0_norm2_weight: "f32[128]", p_core_ring_encoder_layers_0_norm2_bias: "f32[128]", p_core_mol_encoder_layers_0_self_attn_in_proj_weight: "f32[384, 128]", p_core_mol_encoder_layers_0_self_attn_in_proj_bias: "f32[384]", p_core_mol_encoder_layers_0_self_attn_out_proj_weight: "f32[128, 128]", p_core_mol_encoder_layers_0_self_attn_out_proj_bias: "f32[128]", p_core_mol_encoder_layers_0_linear1_weight: "f32[1024, 128]", p_core_mol_encoder_layers_0_linear1_bias: "f32[1024]", p_core_mol_encoder_layers_0_linear2_weight: "f32[128, 1024]", p_core_mol_encoder_layers_0_linear2_bias: "f32[128]", p_core_mol_encoder_layers_0_norm1_weight: "f32[128]", p_core_mol_encoder_layers_0_norm1_bias: "f32[128]", p_core_mol_encoder_layers_0_norm2_weight: "f32[128]", p_core_mol_encoder_layers_0_norm2_bias: "f32[128]", p_core_mol_encoder_layers_1_self_attn_in_proj_weight: "f32[384, 128]", p_core_mol_encoder_layers_1_self_attn_in_proj_bias: "f32[384]", p_core_mol_encoder_layers_1_self_attn_out_proj_weight: "f32[128, 128]", p_core_mol_encoder_layers_1_self_attn_out_proj_bias: "f32[128]", p_core_mol_encoder_layers_1_linear1_weight: "f32[1024, 128]", p_core_mol_encoder_layers_1_linear1_bias: "f32[1024]", p_core_mol_encoder_layers_1_linear2_weight: "f32[128, 1024]", p_core_mol_encoder_layers_1_linear2_bias: "f32[128]", p_core_mol_encoder_layers_1_norm1_weight: "f32[128]", p_core_mol_encoder_layers_1_norm1_bias: "f32[128]", p_core_mol_encoder_layers_1_norm2_weight: "f32[128]", p_core_mol_encoder_layers_1_norm2_bias: "f32[128]", p_core_mol_encoder_layers_2_self_attn_in_proj_weight: "f32[384, 128]", p_core_mol_encoder_layers_2_self_attn_in_proj_bias: "f32[384]", p_core_mol_encoder_layers_2_self_attn_out_proj_weight: "f32[128, 128]", p_core_mol_encoder_layers_2_self_attn_out_proj_bias: "f32[128]", p_core_mol_encoder_layers_2_linear1_weight: "f32[1024, 128]", p_core_mol_encoder_layers_2_linear1_bias: "f32[1024]", p_core_mol_encoder_layers_2_linear2_weight: "f32[128, 1024]", p_core_mol_encoder_layers_2_linear2_bias: "f32[128]", p_core_mol_encoder_layers_2_norm1_weight: "f32[128]", p_core_mol_encoder_layers_2_norm1_bias: "f32[128]", p_core_mol_encoder_layers_2_norm2_weight: "f32[128]", p_core_mol_encoder_layers_2_norm2_bias: "f32[128]", p_core_mol_encoder_layers_3_self_attn_in_proj_weight: "f32[384, 128]", p_core_mol_encoder_layers_3_self_attn_in_proj_bias: "f32[384]", p_core_mol_encoder_layers_3_self_attn_out_proj_weight: "f32[128, 128]", p_core_mol_encoder_layers_3_self_attn_out_proj_bias: "f32[128]", p_core_mol_encoder_layers_3_linear1_weight: "f32[1024, 128]", p_core_mol_encoder_layers_3_linear1_bias: "f32[1024]", p_core_mol_encoder_layers_3_linear2_weight: "f32[128, 1024]", p_core_mol_encoder_layers_3_linear2_bias: "f32[128]", p_core_mol_encoder_layers_3_norm1_weight: "f32[128]", p_core_mol_encoder_layers_3_norm1_bias: "f32[128]", p_core_mol_encoder_layers_3_norm2_weight: "f32[128]", p_core_mol_encoder_layers_3_norm2_bias: "f32[128]", p_predictors_hidden_layers_lins_0_weight: "f32[128, 128]", p_predictors_hidden_layers_lins_0_bias: "f32[128]", p_predictors_out_layer_weight: "f32[1, 128]", p_predictors_out_layer_bias: "f32[1]", b_core_node_processor_norm_running_mean: "f32[128]", b_core_node_processor_norm_running_var: "f32[128]", b_core_node_processor_norm_num_batches_tracked: "i64[]", x: "f32[s0, 128]", padded_xr: "f32[128, 64, 128]", rings_mask: "b8[128, 64]", cbond_index: "i64[2, s1]"):
             # 
            sym_size_int_17: "Sym(s0)" = torch.ops.aten.sym_size.int(x, 0)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:414 in forward, code: src_key_padding_mask = F._canonical_mask(
            zeros_like: "f32[128, 64]" = torch.ops.aten.zeros_like.default(rings_mask, dtype = torch.float32, pin_memory = False)
            masked_fill: "f32[128, 64]" = torch.ops.aten.masked_fill.Scalar(zeros_like, rings_mask, -inf);  zeros_like = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1339 in forward, code: query = key = value = query.transpose(1, 0)
            transpose: "f32[64, 128, 128]" = torch.ops.aten.transpose.int(padded_xr, 1, 0)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1373 in forward, code: attn_output, attn_output_weights = F.multi_head_attention_forward(
            linear: "f32[64, 128, 384]" = torch.ops.aten.linear.default(transpose, p_core_ring_encoder_layers_0_self_attn_in_proj_weight, p_core_ring_encoder_layers_0_self_attn_in_proj_bias);  transpose = p_core_ring_encoder_layers_0_self_attn_in_proj_weight = p_core_ring_encoder_layers_0_self_attn_in_proj_bias = None
            view: "f32[64, 128, 3, 128]" = torch.ops.aten.view.default(linear, [64, 128, 3, 128]);  linear = None
            unsqueeze: "f32[1, 64, 128, 3, 128]" = torch.ops.aten.unsqueeze.default(view, 0);  view = None
            transpose_1: "f32[3, 64, 128, 1, 128]" = torch.ops.aten.transpose.int(unsqueeze, 0, -2);  unsqueeze = None
            squeeze: "f32[3, 64, 128, 128]" = torch.ops.aten.squeeze.dim(transpose_1, -2);  transpose_1 = None
            clone: "f32[3, 64, 128, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
            select: "f32[64, 128, 128]" = torch.ops.aten.select.int(clone, 0, 0)
            select_1: "f32[64, 128, 128]" = torch.ops.aten.select.int(clone, 0, 1)
            select_2: "f32[64, 128, 128]" = torch.ops.aten.select.int(clone, 0, 2);  clone = None
            view_1: "f32[64, 256, 64]" = torch.ops.aten.view.default(select, [64, 256, 64]);  select = None
            transpose_2: "f32[256, 64, 64]" = torch.ops.aten.transpose.int(view_1, 0, 1);  view_1 = None
            view_2: "f32[64, 256, 64]" = torch.ops.aten.view.default(select_1, [64, 256, 64]);  select_1 = None
            transpose_3: "f32[256, 64, 64]" = torch.ops.aten.transpose.int(view_2, 0, 1);  view_2 = None
            view_3: "f32[64, 256, 64]" = torch.ops.aten.view.default(select_2, [64, 256, 64]);  select_2 = None
            transpose_4: "f32[256, 64, 64]" = torch.ops.aten.transpose.int(view_3, 0, 1);  view_3 = None
            view_5: "f32[128, 1, 1, 64]" = torch.ops.aten.view.default(masked_fill, [128, 1, 1, 64]);  masked_fill = None
            expand_1: "f32[128, 2, 1, 64]" = torch.ops.aten.expand.default(view_5, [-1, 2, -1, -1]);  view_5 = None
            clone_1: "f32[128, 2, 1, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
            _unsafe_view: "f32[256, 1, 64]" = torch.ops.aten._unsafe_view.default(clone_1, [256, 1, 64]);  clone_1 = None
            view_6: "f32[128, 2, 1, 64]" = torch.ops.aten.view.default(_unsafe_view, [128, 2, -1, 64]);  _unsafe_view = None
            view_7: "f32[128, 2, 64, 64]" = torch.ops.aten.view.default(transpose_2, [128, 2, 64, 64]);  transpose_2 = None
            view_8: "f32[128, 2, 64, 64]" = torch.ops.aten.view.default(transpose_3, [128, 2, 64, 64]);  transpose_3 = None
            view_9: "f32[128, 2, 64, 64]" = torch.ops.aten.view.default(transpose_4, [128, 2, 64, 64]);  transpose_4 = None
            scaled_dot_product_attention: "f32[128, 2, 64, 64]" = torch.ops.aten.scaled_dot_product_attention.default(view_7, view_8, view_9, view_6);  view_7 = view_8 = view_9 = view_6 = None
            permute: "f32[64, 128, 2, 64]" = torch.ops.aten.permute.default(scaled_dot_product_attention, [2, 0, 1, 3]);  scaled_dot_product_attention = None
            view_10: "f32[8192, 128]" = torch.ops.aten.view.default(permute, [8192, 128]);  permute = None
            linear_1: "f32[8192, 128]" = torch.ops.aten.linear.default(view_10, p_core_ring_encoder_layers_0_self_attn_out_proj_weight, p_core_ring_encoder_layers_0_self_attn_out_proj_bias);  view_10 = p_core_ring_encoder_layers_0_self_attn_out_proj_weight = p_core_ring_encoder_layers_0_self_attn_out_proj_bias = None
            view_11: "f32[64, 128, 128]" = torch.ops.aten.view.default(linear_1, [64, 128, 128]);  linear_1 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1395 in forward, code: return attn_output.transpose(1, 0), attn_output_weights
            transpose_5: "f32[128, 64, 128]" = torch.ops.aten.transpose.int(view_11, 1, 0);  view_11 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_2: "f32[128, 64, 128]" = torch.ops.aten.clone.default(transpose_5);  transpose_5 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:919 in forward, code: x
            add: "f32[128, 64, 128]" = torch.ops.aten.add.Tensor(padded_xr, clone_2);  padded_xr = clone_2 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/normalization.py:217 in forward, code: return F.layer_norm(
            layer_norm: "f32[128, 64, 128]" = torch.ops.aten.layer_norm.default(add, [128], p_core_ring_encoder_layers_0_norm1_weight, p_core_ring_encoder_layers_0_norm1_bias);  add = p_core_ring_encoder_layers_0_norm1_weight = p_core_ring_encoder_layers_0_norm1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_2: "f32[128, 64, 1024]" = torch.ops.aten.linear.default(layer_norm, p_core_ring_encoder_layers_0_linear1_weight, p_core_ring_encoder_layers_0_linear1_bias);  p_core_ring_encoder_layers_0_linear1_weight = p_core_ring_encoder_layers_0_linear1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:922 in forward, code: x = self.norm2(x + self._ff_block(x))
            relu: "f32[128, 64, 1024]" = torch.ops.aten.relu.default(linear_2);  linear_2 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_3: "f32[128, 64, 1024]" = torch.ops.aten.clone.default(relu);  relu = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_3: "f32[128, 64, 128]" = torch.ops.aten.linear.default(clone_3, p_core_ring_encoder_layers_0_linear2_weight, p_core_ring_encoder_layers_0_linear2_bias);  clone_3 = p_core_ring_encoder_layers_0_linear2_weight = p_core_ring_encoder_layers_0_linear2_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_4: "f32[128, 64, 128]" = torch.ops.aten.clone.default(linear_3);  linear_3 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:922 in forward, code: x = self.norm2(x + self._ff_block(x))
            add_1: "f32[128, 64, 128]" = torch.ops.aten.add.Tensor(layer_norm, clone_4);  layer_norm = clone_4 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/normalization.py:217 in forward, code: return F.layer_norm(
            layer_norm_1: "f32[128, 64, 128]" = torch.ops.aten.layer_norm.default(add_1, [128], p_core_ring_encoder_layers_0_norm2_weight, p_core_ring_encoder_layers_0_norm2_bias);  add_1 = p_core_ring_encoder_layers_0_norm2_weight = p_core_ring_encoder_layers_0_norm2_bias = None
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/infer/infer_models/cbond_infer_model.py:34 in forward, code: xr = self.core.f_rings(padded_Xr, rings_mask)
            unsqueeze_1: "b8[128, 64, 1]" = torch.ops.aten.unsqueeze.default(rings_mask, -1)
            masked_fill_1: "f32[128, 64, 128]" = torch.ops.aten.masked_fill.Scalar(layer_norm_1, unsqueeze_1, 0.0);  layer_norm_1 = unsqueeze_1 = None
            abs_1: "f32[128, 64, 128]" = torch.ops.aten.abs.default(masked_fill_1)
            argmax: "i64[128, 128]" = torch.ops.aten.argmax.default(abs_1, -2);  abs_1 = None
            unsqueeze_2: "i64[128, 1, 128]" = torch.ops.aten.unsqueeze.default(argmax, -2);  argmax = None
            gather: "f32[128, 1, 128]" = torch.ops.aten.gather.default(masked_fill_1, -2, unsqueeze_2);  masked_fill_1 = unsqueeze_2 = None
            squeeze_1: "f32[128, 128]" = torch.ops.aten.squeeze.dim(gather, -2);  gather = None
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/infer/infer_models/cbond_infer_model.py:35 in forward, code: xr_mask = torch.all(rings_mask, dim=1)
            all_1: "b8[128]" = torch.ops.aten.all.dim(rings_mask, 1);  rings_mask = None
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/infer/infer_models/cbond_infer_model.py:36 in forward, code: seq = self.core._mol_attention(x, xr, xr_mask)
            cat: "f32[s0 + 131, 128]" = torch.ops.aten.cat.default([p_core_cls, x, p_core_ring, squeeze_1, p_core_end]);  p_core_cls = x = p_core_ring = squeeze_1 = p_core_end = None
            add_613: "Sym(s0 + 131)" = 131 + sym_size_int_17
            unsqueeze_3: "f32[1, s0 + 131, 128]" = torch.ops.aten.unsqueeze.default(cat, 0);  cat = None
            add_8: "Sym(s0 + 2)" = 2 + sym_size_int_17
            zeros: "b8[s0 + 2]" = torch.ops.aten.zeros.default([add_8], dtype = torch.bool, device = device(type='cpu'), pin_memory = False);  add_8 = None
            zeros_1: "b8[1]" = torch.ops.aten.zeros.default([1], dtype = torch.bool, device = device(type='cpu'), pin_memory = False)
            cat_1: "b8[s0 + 131]" = torch.ops.aten.cat.default([zeros, all_1, zeros_1]);  zeros = all_1 = zeros_1 = None
            unsqueeze_4: "b8[1, s0 + 131]" = torch.ops.aten.unsqueeze.default(cat_1, 0);  cat_1 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:414 in forward, code: src_key_padding_mask = F._canonical_mask(
            zeros_like_1: "f32[1, s0 + 131]" = torch.ops.aten.zeros_like.default(unsqueeze_4, dtype = torch.float32, pin_memory = False)
            masked_fill_2: "f32[1, s0 + 131]" = torch.ops.aten.masked_fill.Scalar(zeros_like_1, unsqueeze_4, -inf);  zeros_like_1 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1339 in forward, code: query = key = value = query.transpose(1, 0)
            transpose_6: "f32[s0 + 131, 1, 128]" = torch.ops.aten.transpose.int(unsqueeze_3, 1, 0)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1373 in forward, code: attn_output, attn_output_weights = F.multi_head_attention_forward(
            linear_4: "f32[s0 + 131, 1, 384]" = torch.ops.aten.linear.default(transpose_6, p_core_mol_encoder_layers_0_self_attn_in_proj_weight, p_core_mol_encoder_layers_0_self_attn_in_proj_bias);  transpose_6 = p_core_mol_encoder_layers_0_self_attn_in_proj_weight = p_core_mol_encoder_layers_0_self_attn_in_proj_bias = None
            view_12: "f32[s0 + 131, 1, 3, 128]" = torch.ops.aten.view.default(linear_4, [add_613, 1, 3, 128]);  linear_4 = None
            unsqueeze_5: "f32[1, s0 + 131, 1, 3, 128]" = torch.ops.aten.unsqueeze.default(view_12, 0);  view_12 = None
            transpose_7: "f32[3, s0 + 131, 1, 1, 128]" = torch.ops.aten.transpose.int(unsqueeze_5, 0, -2);  unsqueeze_5 = None
            squeeze_2: "f32[3, s0 + 131, 1, 128]" = torch.ops.aten.squeeze.dim(transpose_7, -2);  transpose_7 = None
            clone_5: "f32[3, s0 + 131, 1, 128]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
            select_3: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_5, 0, 0)
            select_4: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_5, 0, 1)
            select_5: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_5, 0, 2);  clone_5 = None
            view_13: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_3, [add_613, 4, 32]);  select_3 = None
            transpose_8: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_13, 0, 1);  view_13 = None
            view_14: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_4, [add_613, 4, 32]);  select_4 = None
            transpose_9: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_14, 0, 1);  view_14 = None
            view_15: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_5, [add_613, 4, 32]);  select_5 = None
            transpose_10: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_15, 0, 1);  view_15 = None
            view_19: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_8, [1, 4, add_613, 32]);  transpose_8 = None
            view_20: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_9, [1, 4, add_613, 32]);  transpose_9 = None
            view_21: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_10, [1, 4, add_613, 32]);  transpose_10 = None
            view_22: "f32[1, 1, 1, s0 + 131]" = torch.ops.aten.view.default(masked_fill_2, [1, 1, 1, add_613])
            expand_3: "f32[1, 4, 1, s0 + 131]" = torch.ops.aten.expand.default(view_22, [-1, 4, -1, -1]);  view_22 = None
            view_23: "f32[4, 1, s0 + 131]" = torch.ops.aten.view.default(expand_3, [4, 1, add_613]);  expand_3 = None
            view_24: "f32[1, 4, 1, s0 + 131]" = torch.ops.aten.view.default(view_23, [1, 4, -1, add_613]);  view_23 = None
            scaled_dot_product_attention_1: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.scaled_dot_product_attention.default(view_19, view_20, view_21, view_24);  view_19 = view_20 = view_21 = view_24 = None
            permute_1: "f32[s0 + 131, 1, 4, 32]" = torch.ops.aten.permute.default(scaled_dot_product_attention_1, [2, 0, 1, 3]);  scaled_dot_product_attention_1 = None
            view_25: "f32[s0 + 131, 128]" = torch.ops.aten.view.default(permute_1, [add_613, 128]);  permute_1 = None
            linear_5: "f32[s0 + 131, 128]" = torch.ops.aten.linear.default(view_25, p_core_mol_encoder_layers_0_self_attn_out_proj_weight, p_core_mol_encoder_layers_0_self_attn_out_proj_bias);  view_25 = p_core_mol_encoder_layers_0_self_attn_out_proj_weight = p_core_mol_encoder_layers_0_self_attn_out_proj_bias = None
            view_26: "f32[s0 + 131, 1, 128]" = torch.ops.aten.view.default(linear_5, [add_613, 1, 128]);  linear_5 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1395 in forward, code: return attn_output.transpose(1, 0), attn_output_weights
            transpose_11: "f32[1, s0 + 131, 128]" = torch.ops.aten.transpose.int(view_26, 1, 0);  view_26 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_6: "f32[1, s0 + 131, 128]" = torch.ops.aten.clone.default(transpose_11);  transpose_11 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:919 in forward, code: x
            add_130: "f32[1, s0 + 131, 128]" = torch.ops.aten.add.Tensor(unsqueeze_3, clone_6);  unsqueeze_3 = clone_6 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/normalization.py:217 in forward, code: return F.layer_norm(
            layer_norm_2: "f32[1, s0 + 131, 128]" = torch.ops.aten.layer_norm.default(add_130, [128], p_core_mol_encoder_layers_0_norm1_weight, p_core_mol_encoder_layers_0_norm1_bias);  add_130 = p_core_mol_encoder_layers_0_norm1_weight = p_core_mol_encoder_layers_0_norm1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_6: "f32[1, s0 + 131, 1024]" = torch.ops.aten.linear.default(layer_norm_2, p_core_mol_encoder_layers_0_linear1_weight, p_core_mol_encoder_layers_0_linear1_bias);  p_core_mol_encoder_layers_0_linear1_weight = p_core_mol_encoder_layers_0_linear1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:922 in forward, code: x = self.norm2(x + self._ff_block(x))
            relu_1: "f32[1, s0 + 131, 1024]" = torch.ops.aten.relu.default(linear_6);  linear_6 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_7: "f32[1, s0 + 131, 1024]" = torch.ops.aten.clone.default(relu_1);  relu_1 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_7: "f32[1, s0 + 131, 128]" = torch.ops.aten.linear.default(clone_7, p_core_mol_encoder_layers_0_linear2_weight, p_core_mol_encoder_layers_0_linear2_bias);  clone_7 = p_core_mol_encoder_layers_0_linear2_weight = p_core_mol_encoder_layers_0_linear2_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_8: "f32[1, s0 + 131, 128]" = torch.ops.aten.clone.default(linear_7);  linear_7 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:922 in forward, code: x = self.norm2(x + self._ff_block(x))
            add_152: "f32[1, s0 + 131, 128]" = torch.ops.aten.add.Tensor(layer_norm_2, clone_8);  layer_norm_2 = clone_8 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/normalization.py:217 in forward, code: return F.layer_norm(
            layer_norm_3: "f32[1, s0 + 131, 128]" = torch.ops.aten.layer_norm.default(add_152, [128], p_core_mol_encoder_layers_0_norm2_weight, p_core_mol_encoder_layers_0_norm2_bias);  add_152 = p_core_mol_encoder_layers_0_norm2_weight = p_core_mol_encoder_layers_0_norm2_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1339 in forward, code: query = key = value = query.transpose(1, 0)
            transpose_12: "f32[s0 + 131, 1, 128]" = torch.ops.aten.transpose.int(layer_norm_3, 1, 0)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1373 in forward, code: attn_output, attn_output_weights = F.multi_head_attention_forward(
            linear_8: "f32[s0 + 131, 1, 384]" = torch.ops.aten.linear.default(transpose_12, p_core_mol_encoder_layers_1_self_attn_in_proj_weight, p_core_mol_encoder_layers_1_self_attn_in_proj_bias);  transpose_12 = p_core_mol_encoder_layers_1_self_attn_in_proj_weight = p_core_mol_encoder_layers_1_self_attn_in_proj_bias = None
            view_27: "f32[s0 + 131, 1, 3, 128]" = torch.ops.aten.view.default(linear_8, [add_613, 1, 3, 128]);  linear_8 = None
            unsqueeze_6: "f32[1, s0 + 131, 1, 3, 128]" = torch.ops.aten.unsqueeze.default(view_27, 0);  view_27 = None
            transpose_13: "f32[3, s0 + 131, 1, 1, 128]" = torch.ops.aten.transpose.int(unsqueeze_6, 0, -2);  unsqueeze_6 = None
            squeeze_3: "f32[3, s0 + 131, 1, 128]" = torch.ops.aten.squeeze.dim(transpose_13, -2);  transpose_13 = None
            clone_9: "f32[3, s0 + 131, 1, 128]" = torch.ops.aten.clone.default(squeeze_3, memory_format = torch.contiguous_format);  squeeze_3 = None
            select_6: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_9, 0, 0)
            select_7: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_9, 0, 1)
            select_8: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_9, 0, 2);  clone_9 = None
            view_28: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_6, [add_613, 4, 32]);  select_6 = None
            transpose_14: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_28, 0, 1);  view_28 = None
            view_29: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_7, [add_613, 4, 32]);  select_7 = None
            transpose_15: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_29, 0, 1);  view_29 = None
            view_30: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_8, [add_613, 4, 32]);  select_8 = None
            transpose_16: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_30, 0, 1);  view_30 = None
            view_34: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_14, [1, 4, add_613, 32]);  transpose_14 = None
            view_35: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_15, [1, 4, add_613, 32]);  transpose_15 = None
            view_36: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_16, [1, 4, add_613, 32]);  transpose_16 = None
            view_37: "f32[1, 1, 1, s0 + 131]" = torch.ops.aten.view.default(masked_fill_2, [1, 1, 1, add_613])
            expand_5: "f32[1, 4, 1, s0 + 131]" = torch.ops.aten.expand.default(view_37, [-1, 4, -1, -1]);  view_37 = None
            view_38: "f32[4, 1, s0 + 131]" = torch.ops.aten.view.default(expand_5, [4, 1, add_613]);  expand_5 = None
            view_39: "f32[1, 4, 1, s0 + 131]" = torch.ops.aten.view.default(view_38, [1, 4, -1, add_613]);  view_38 = None
            scaled_dot_product_attention_2: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.scaled_dot_product_attention.default(view_34, view_35, view_36, view_39);  view_34 = view_35 = view_36 = view_39 = None
            permute_2: "f32[s0 + 131, 1, 4, 32]" = torch.ops.aten.permute.default(scaled_dot_product_attention_2, [2, 0, 1, 3]);  scaled_dot_product_attention_2 = None
            view_40: "f32[s0 + 131, 128]" = torch.ops.aten.view.default(permute_2, [add_613, 128]);  permute_2 = None
            linear_9: "f32[s0 + 131, 128]" = torch.ops.aten.linear.default(view_40, p_core_mol_encoder_layers_1_self_attn_out_proj_weight, p_core_mol_encoder_layers_1_self_attn_out_proj_bias);  view_40 = p_core_mol_encoder_layers_1_self_attn_out_proj_weight = p_core_mol_encoder_layers_1_self_attn_out_proj_bias = None
            view_41: "f32[s0 + 131, 1, 128]" = torch.ops.aten.view.default(linear_9, [add_613, 1, 128]);  linear_9 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1395 in forward, code: return attn_output.transpose(1, 0), attn_output_weights
            transpose_17: "f32[1, s0 + 131, 128]" = torch.ops.aten.transpose.int(view_41, 1, 0);  view_41 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_10: "f32[1, s0 + 131, 128]" = torch.ops.aten.clone.default(transpose_17);  transpose_17 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:919 in forward, code: x
            add_265: "f32[1, s0 + 131, 128]" = torch.ops.aten.add.Tensor(layer_norm_3, clone_10);  layer_norm_3 = clone_10 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/normalization.py:217 in forward, code: return F.layer_norm(
            layer_norm_4: "f32[1, s0 + 131, 128]" = torch.ops.aten.layer_norm.default(add_265, [128], p_core_mol_encoder_layers_1_norm1_weight, p_core_mol_encoder_layers_1_norm1_bias);  add_265 = p_core_mol_encoder_layers_1_norm1_weight = p_core_mol_encoder_layers_1_norm1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_10: "f32[1, s0 + 131, 1024]" = torch.ops.aten.linear.default(layer_norm_4, p_core_mol_encoder_layers_1_linear1_weight, p_core_mol_encoder_layers_1_linear1_bias);  p_core_mol_encoder_layers_1_linear1_weight = p_core_mol_encoder_layers_1_linear1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:922 in forward, code: x = self.norm2(x + self._ff_block(x))
            relu_2: "f32[1, s0 + 131, 1024]" = torch.ops.aten.relu.default(linear_10);  linear_10 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_11: "f32[1, s0 + 131, 1024]" = torch.ops.aten.clone.default(relu_2);  relu_2 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_11: "f32[1, s0 + 131, 128]" = torch.ops.aten.linear.default(clone_11, p_core_mol_encoder_layers_1_linear2_weight, p_core_mol_encoder_layers_1_linear2_bias);  clone_11 = p_core_mol_encoder_layers_1_linear2_weight = p_core_mol_encoder_layers_1_linear2_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_12: "f32[1, s0 + 131, 128]" = torch.ops.aten.clone.default(linear_11);  linear_11 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:922 in forward, code: x = self.norm2(x + self._ff_block(x))
            add_287: "f32[1, s0 + 131, 128]" = torch.ops.aten.add.Tensor(layer_norm_4, clone_12);  layer_norm_4 = clone_12 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/normalization.py:217 in forward, code: return F.layer_norm(
            layer_norm_5: "f32[1, s0 + 131, 128]" = torch.ops.aten.layer_norm.default(add_287, [128], p_core_mol_encoder_layers_1_norm2_weight, p_core_mol_encoder_layers_1_norm2_bias);  add_287 = p_core_mol_encoder_layers_1_norm2_weight = p_core_mol_encoder_layers_1_norm2_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1339 in forward, code: query = key = value = query.transpose(1, 0)
            transpose_18: "f32[s0 + 131, 1, 128]" = torch.ops.aten.transpose.int(layer_norm_5, 1, 0)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1373 in forward, code: attn_output, attn_output_weights = F.multi_head_attention_forward(
            linear_12: "f32[s0 + 131, 1, 384]" = torch.ops.aten.linear.default(transpose_18, p_core_mol_encoder_layers_2_self_attn_in_proj_weight, p_core_mol_encoder_layers_2_self_attn_in_proj_bias);  transpose_18 = p_core_mol_encoder_layers_2_self_attn_in_proj_weight = p_core_mol_encoder_layers_2_self_attn_in_proj_bias = None
            view_42: "f32[s0 + 131, 1, 3, 128]" = torch.ops.aten.view.default(linear_12, [add_613, 1, 3, 128]);  linear_12 = None
            unsqueeze_7: "f32[1, s0 + 131, 1, 3, 128]" = torch.ops.aten.unsqueeze.default(view_42, 0);  view_42 = None
            transpose_19: "f32[3, s0 + 131, 1, 1, 128]" = torch.ops.aten.transpose.int(unsqueeze_7, 0, -2);  unsqueeze_7 = None
            squeeze_4: "f32[3, s0 + 131, 1, 128]" = torch.ops.aten.squeeze.dim(transpose_19, -2);  transpose_19 = None
            clone_13: "f32[3, s0 + 131, 1, 128]" = torch.ops.aten.clone.default(squeeze_4, memory_format = torch.contiguous_format);  squeeze_4 = None
            select_9: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_13, 0, 0)
            select_10: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_13, 0, 1)
            select_11: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_13, 0, 2);  clone_13 = None
            view_43: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_9, [add_613, 4, 32]);  select_9 = None
            transpose_20: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_43, 0, 1);  view_43 = None
            view_44: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_10, [add_613, 4, 32]);  select_10 = None
            transpose_21: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_44, 0, 1);  view_44 = None
            view_45: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_11, [add_613, 4, 32]);  select_11 = None
            transpose_22: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_45, 0, 1);  view_45 = None
            view_49: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_20, [1, 4, add_613, 32]);  transpose_20 = None
            view_50: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_21, [1, 4, add_613, 32]);  transpose_21 = None
            view_51: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_22, [1, 4, add_613, 32]);  transpose_22 = None
            view_52: "f32[1, 1, 1, s0 + 131]" = torch.ops.aten.view.default(masked_fill_2, [1, 1, 1, add_613])
            expand_7: "f32[1, 4, 1, s0 + 131]" = torch.ops.aten.expand.default(view_52, [-1, 4, -1, -1]);  view_52 = None
            view_53: "f32[4, 1, s0 + 131]" = torch.ops.aten.view.default(expand_7, [4, 1, add_613]);  expand_7 = None
            view_54: "f32[1, 4, 1, s0 + 131]" = torch.ops.aten.view.default(view_53, [1, 4, -1, add_613]);  view_53 = None
            scaled_dot_product_attention_3: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.scaled_dot_product_attention.default(view_49, view_50, view_51, view_54);  view_49 = view_50 = view_51 = view_54 = None
            permute_3: "f32[s0 + 131, 1, 4, 32]" = torch.ops.aten.permute.default(scaled_dot_product_attention_3, [2, 0, 1, 3]);  scaled_dot_product_attention_3 = None
            view_55: "f32[s0 + 131, 128]" = torch.ops.aten.view.default(permute_3, [add_613, 128]);  permute_3 = None
            linear_13: "f32[s0 + 131, 128]" = torch.ops.aten.linear.default(view_55, p_core_mol_encoder_layers_2_self_attn_out_proj_weight, p_core_mol_encoder_layers_2_self_attn_out_proj_bias);  view_55 = p_core_mol_encoder_layers_2_self_attn_out_proj_weight = p_core_mol_encoder_layers_2_self_attn_out_proj_bias = None
            view_56: "f32[s0 + 131, 1, 128]" = torch.ops.aten.view.default(linear_13, [add_613, 1, 128]);  linear_13 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1395 in forward, code: return attn_output.transpose(1, 0), attn_output_weights
            transpose_23: "f32[1, s0 + 131, 128]" = torch.ops.aten.transpose.int(view_56, 1, 0);  view_56 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_14: "f32[1, s0 + 131, 128]" = torch.ops.aten.clone.default(transpose_23);  transpose_23 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:919 in forward, code: x
            add_400: "f32[1, s0 + 131, 128]" = torch.ops.aten.add.Tensor(layer_norm_5, clone_14);  layer_norm_5 = clone_14 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/normalization.py:217 in forward, code: return F.layer_norm(
            layer_norm_6: "f32[1, s0 + 131, 128]" = torch.ops.aten.layer_norm.default(add_400, [128], p_core_mol_encoder_layers_2_norm1_weight, p_core_mol_encoder_layers_2_norm1_bias);  add_400 = p_core_mol_encoder_layers_2_norm1_weight = p_core_mol_encoder_layers_2_norm1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_14: "f32[1, s0 + 131, 1024]" = torch.ops.aten.linear.default(layer_norm_6, p_core_mol_encoder_layers_2_linear1_weight, p_core_mol_encoder_layers_2_linear1_bias);  p_core_mol_encoder_layers_2_linear1_weight = p_core_mol_encoder_layers_2_linear1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:922 in forward, code: x = self.norm2(x + self._ff_block(x))
            relu_3: "f32[1, s0 + 131, 1024]" = torch.ops.aten.relu.default(linear_14);  linear_14 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_15: "f32[1, s0 + 131, 1024]" = torch.ops.aten.clone.default(relu_3);  relu_3 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_15: "f32[1, s0 + 131, 128]" = torch.ops.aten.linear.default(clone_15, p_core_mol_encoder_layers_2_linear2_weight, p_core_mol_encoder_layers_2_linear2_bias);  clone_15 = p_core_mol_encoder_layers_2_linear2_weight = p_core_mol_encoder_layers_2_linear2_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_16: "f32[1, s0 + 131, 128]" = torch.ops.aten.clone.default(linear_15);  linear_15 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:922 in forward, code: x = self.norm2(x + self._ff_block(x))
            add_422: "f32[1, s0 + 131, 128]" = torch.ops.aten.add.Tensor(layer_norm_6, clone_16);  layer_norm_6 = clone_16 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/normalization.py:217 in forward, code: return F.layer_norm(
            layer_norm_7: "f32[1, s0 + 131, 128]" = torch.ops.aten.layer_norm.default(add_422, [128], p_core_mol_encoder_layers_2_norm2_weight, p_core_mol_encoder_layers_2_norm2_bias);  add_422 = p_core_mol_encoder_layers_2_norm2_weight = p_core_mol_encoder_layers_2_norm2_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1339 in forward, code: query = key = value = query.transpose(1, 0)
            transpose_24: "f32[s0 + 131, 1, 128]" = torch.ops.aten.transpose.int(layer_norm_7, 1, 0)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1373 in forward, code: attn_output, attn_output_weights = F.multi_head_attention_forward(
            linear_16: "f32[s0 + 131, 1, 384]" = torch.ops.aten.linear.default(transpose_24, p_core_mol_encoder_layers_3_self_attn_in_proj_weight, p_core_mol_encoder_layers_3_self_attn_in_proj_bias);  transpose_24 = p_core_mol_encoder_layers_3_self_attn_in_proj_weight = p_core_mol_encoder_layers_3_self_attn_in_proj_bias = None
            view_57: "f32[s0 + 131, 1, 3, 128]" = torch.ops.aten.view.default(linear_16, [add_613, 1, 3, 128]);  linear_16 = None
            unsqueeze_8: "f32[1, s0 + 131, 1, 3, 128]" = torch.ops.aten.unsqueeze.default(view_57, 0);  view_57 = None
            transpose_25: "f32[3, s0 + 131, 1, 1, 128]" = torch.ops.aten.transpose.int(unsqueeze_8, 0, -2);  unsqueeze_8 = None
            squeeze_5: "f32[3, s0 + 131, 1, 128]" = torch.ops.aten.squeeze.dim(transpose_25, -2);  transpose_25 = None
            clone_17: "f32[3, s0 + 131, 1, 128]" = torch.ops.aten.clone.default(squeeze_5, memory_format = torch.contiguous_format);  squeeze_5 = None
            select_12: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_17, 0, 0)
            select_13: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_17, 0, 1)
            select_14: "f32[s0 + 131, 1, 128]" = torch.ops.aten.select.int(clone_17, 0, 2);  clone_17 = None
            view_58: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_12, [add_613, 4, 32]);  select_12 = None
            transpose_26: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_58, 0, 1);  view_58 = None
            view_59: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_13, [add_613, 4, 32]);  select_13 = None
            transpose_27: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_59, 0, 1);  view_59 = None
            view_60: "f32[s0 + 131, 4, 32]" = torch.ops.aten.view.default(select_14, [add_613, 4, 32]);  select_14 = None
            transpose_28: "f32[4, s0 + 131, 32]" = torch.ops.aten.transpose.int(view_60, 0, 1);  view_60 = None
            view_64: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_26, [1, 4, add_613, 32]);  transpose_26 = None
            view_65: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_27, [1, 4, add_613, 32]);  transpose_27 = None
            view_66: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.view.default(transpose_28, [1, 4, add_613, 32]);  transpose_28 = None
            view_67: "f32[1, 1, 1, s0 + 131]" = torch.ops.aten.view.default(masked_fill_2, [1, 1, 1, add_613]);  masked_fill_2 = None
            expand_9: "f32[1, 4, 1, s0 + 131]" = torch.ops.aten.expand.default(view_67, [-1, 4, -1, -1]);  view_67 = None
            view_68: "f32[4, 1, s0 + 131]" = torch.ops.aten.view.default(expand_9, [4, 1, add_613]);  expand_9 = None
            view_69: "f32[1, 4, 1, s0 + 131]" = torch.ops.aten.view.default(view_68, [1, 4, -1, add_613]);  view_68 = None
            scaled_dot_product_attention_4: "f32[1, 4, s0 + 131, 32]" = torch.ops.aten.scaled_dot_product_attention.default(view_64, view_65, view_66, view_69);  view_64 = view_65 = view_66 = view_69 = None
            permute_4: "f32[s0 + 131, 1, 4, 32]" = torch.ops.aten.permute.default(scaled_dot_product_attention_4, [2, 0, 1, 3]);  scaled_dot_product_attention_4 = None
            view_70: "f32[s0 + 131, 128]" = torch.ops.aten.view.default(permute_4, [add_613, 128]);  permute_4 = None
            linear_17: "f32[s0 + 131, 128]" = torch.ops.aten.linear.default(view_70, p_core_mol_encoder_layers_3_self_attn_out_proj_weight, p_core_mol_encoder_layers_3_self_attn_out_proj_bias);  view_70 = p_core_mol_encoder_layers_3_self_attn_out_proj_weight = p_core_mol_encoder_layers_3_self_attn_out_proj_bias = None
            view_71: "f32[s0 + 131, 1, 128]" = torch.ops.aten.view.default(linear_17, [add_613, 1, 128]);  linear_17 = add_613 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:1395 in forward, code: return attn_output.transpose(1, 0), attn_output_weights
            transpose_29: "f32[1, s0 + 131, 128]" = torch.ops.aten.transpose.int(view_71, 1, 0);  view_71 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_18: "f32[1, s0 + 131, 128]" = torch.ops.aten.clone.default(transpose_29);  transpose_29 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:919 in forward, code: x
            add_535: "f32[1, s0 + 131, 128]" = torch.ops.aten.add.Tensor(layer_norm_7, clone_18);  layer_norm_7 = clone_18 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/normalization.py:217 in forward, code: return F.layer_norm(
            layer_norm_8: "f32[1, s0 + 131, 128]" = torch.ops.aten.layer_norm.default(add_535, [128], p_core_mol_encoder_layers_3_norm1_weight, p_core_mol_encoder_layers_3_norm1_bias);  add_535 = p_core_mol_encoder_layers_3_norm1_weight = p_core_mol_encoder_layers_3_norm1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_18: "f32[1, s0 + 131, 1024]" = torch.ops.aten.linear.default(layer_norm_8, p_core_mol_encoder_layers_3_linear1_weight, p_core_mol_encoder_layers_3_linear1_bias);  p_core_mol_encoder_layers_3_linear1_weight = p_core_mol_encoder_layers_3_linear1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:922 in forward, code: x = self.norm2(x + self._ff_block(x))
            relu_4: "f32[1, s0 + 131, 1024]" = torch.ops.aten.relu.default(linear_18);  linear_18 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_19: "f32[1, s0 + 131, 1024]" = torch.ops.aten.clone.default(relu_4);  relu_4 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_19: "f32[1, s0 + 131, 128]" = torch.ops.aten.linear.default(clone_19, p_core_mol_encoder_layers_3_linear2_weight, p_core_mol_encoder_layers_3_linear2_bias);  clone_19 = p_core_mol_encoder_layers_3_linear2_weight = p_core_mol_encoder_layers_3_linear2_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_20: "f32[1, s0 + 131, 128]" = torch.ops.aten.clone.default(linear_19);  linear_19 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/transformer.py:922 in forward, code: x = self.norm2(x + self._ff_block(x))
            add_557: "f32[1, s0 + 131, 128]" = torch.ops.aten.add.Tensor(layer_norm_8, clone_20);  layer_norm_8 = clone_20 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/normalization.py:217 in forward, code: return F.layer_norm(
            layer_norm_9: "f32[1, s0 + 131, 128]" = torch.ops.aten.layer_norm.default(add_557, [128], p_core_mol_encoder_layers_3_norm2_weight, p_core_mol_encoder_layers_3_norm2_bias);  add_557 = p_core_mol_encoder_layers_3_norm2_weight = p_core_mol_encoder_layers_3_norm2_bias = None
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/infer/infer_models/cbond_infer_model.py:36 in forward, code: seq = self.core._mol_attention(x, xr, xr_mask)
            unsqueeze_9: "b8[1, s0 + 131, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
            masked_fill_3: "f32[1, s0 + 131, 128]" = torch.ops.aten.masked_fill.Scalar(layer_norm_9, unsqueeze_9, 0.0);  layer_norm_9 = unsqueeze_9 = None
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/infer/infer_models/cbond_infer_model.py:38 in forward, code: zx = seq[0, 1:x.shape[0]+1]
            add_576: "Sym(s0 + 1)" = sym_size_int_17 + 1;  sym_size_int_17 = None
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/infer/infer_models/cbond_infer_model.py:40 in forward, code: upper_idx, lower_idx = cbond_index
            unbind = torch.ops.aten.unbind.int(cbond_index);  cbond_index = None
            getitem: "i64[s1]" = unbind[0]
            getitem_1: "i64[s1]" = unbind[1];  unbind = None
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/infer/infer_models/cbond_infer_model.py:42 in forward, code: uzx = zx[upper_idx]
            select_16: "f32[s0 + 131, 128]" = torch.ops.aten.select.int(masked_fill_3, 0, 0);  masked_fill_3 = None
            slice_2: "f32[s0, 128]" = torch.ops.aten.slice.Tensor(select_16, 0, 1, add_576);  select_16 = add_576 = None
            index: "f32[s1, 128]" = torch.ops.aten.index.Tensor(slice_2, [getitem]);  getitem = None
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/infer/infer_models/cbond_infer_model.py:43 in forward, code: lzx = zx[lower_idx]
            index_1: "f32[s1, 128]" = torch.ops.aten.index.Tensor(slice_2, [getitem_1]);  slice_2 = getitem_1 = None
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/infer/infer_models/cbond_infer_model.py:45 in forward, code: vec_cb = (uzx + lzx) / 2  # cbond vector
            add_593: "f32[s1, 128]" = torch.ops.aten.add.Tensor(index, index_1);  index = index_1 = None
            scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(2, dtype = torch.float32)
            div: "f32[s1, 128]" = torch.ops.aten.div.Tensor(add_593, scalar_tensor_default);  add_593 = scalar_tensor_default = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_20: "f32[s1, 128]" = torch.ops.aten.linear.default(div, p_predictors_hidden_layers_lins_0_weight, p_predictors_hidden_layers_lins_0_bias);  p_predictors_hidden_layers_lins_0_weight = p_predictors_hidden_layers_lins_0_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/models/mlp.py:246 in forward, code: x = F.dropout(x, p=self.dropout[-1], training=self.training)
            clone_21: "f32[s1, 128]" = torch.ops.aten.clone.default(linear_20);  linear_20 = None
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/models/predictor.py:45 in forward, code: z = self.hidden_layers(z) + z
            add_606: "f32[s1, 128]" = torch.ops.aten.add.Tensor(clone_21, div);  clone_21 = div = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_21: "f32[s1, 1]" = torch.ops.aten.linear.default(add_606, p_predictors_out_layer_weight, p_predictors_out_layer_bias);  add_606 = p_predictors_out_layer_weight = p_predictors_out_layer_bias = None
            return (linear_21,)
            
Graph signature: ExportGraphSignature(input_specs=[InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_cls'), target='core.CLS', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring'), target='core.RING', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_end'), target='core.END', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_x_emb_weight'), target='core.node_processor.x_emb.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_lin_weight'), target='core.node_processor.lin.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_lin_bias'), target='core.node_processor.lin.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_norm_weight'), target='core.node_processor.norm.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_norm_bias'), target='core.node_processor.norm.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_0_att'), target='core.node_processor.graph.convs.0.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_0_bias'), target='core.node_processor.graph.convs.0.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_0_lin_l_weight'), target='core.node_processor.graph.convs.0.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_0_lin_l_bias'), target='core.node_processor.graph.convs.0.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_0_lin_r_weight'), target='core.node_processor.graph.convs.0.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_0_lin_r_bias'), target='core.node_processor.graph.convs.0.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_0_lin_edge_weight'), target='core.node_processor.graph.convs.0.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_1_att'), target='core.node_processor.graph.convs.1.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_1_bias'), target='core.node_processor.graph.convs.1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_1_lin_l_weight'), target='core.node_processor.graph.convs.1.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_1_lin_l_bias'), target='core.node_processor.graph.convs.1.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_1_lin_r_weight'), target='core.node_processor.graph.convs.1.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_1_lin_r_bias'), target='core.node_processor.graph.convs.1.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_1_lin_edge_weight'), target='core.node_processor.graph.convs.1.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_2_att'), target='core.node_processor.graph.convs.2.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_2_bias'), target='core.node_processor.graph.convs.2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_2_lin_l_weight'), target='core.node_processor.graph.convs.2.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_2_lin_l_bias'), target='core.node_processor.graph.convs.2.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_2_lin_r_weight'), target='core.node_processor.graph.convs.2.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_2_lin_r_bias'), target='core.node_processor.graph.convs.2.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_2_lin_edge_weight'), target='core.node_processor.graph.convs.2.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_3_att'), target='core.node_processor.graph.convs.3.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_3_bias'), target='core.node_processor.graph.convs.3.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_3_lin_l_weight'), target='core.node_processor.graph.convs.3.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_3_lin_l_bias'), target='core.node_processor.graph.convs.3.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_3_lin_r_weight'), target='core.node_processor.graph.convs.3.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_3_lin_r_bias'), target='core.node_processor.graph.convs.3.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_3_lin_edge_weight'), target='core.node_processor.graph.convs.3.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_4_att'), target='core.node_processor.graph.convs.4.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_4_bias'), target='core.node_processor.graph.convs.4.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_4_lin_l_weight'), target='core.node_processor.graph.convs.4.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_4_lin_l_bias'), target='core.node_processor.graph.convs.4.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_4_lin_r_weight'), target='core.node_processor.graph.convs.4.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_4_lin_r_bias'), target='core.node_processor.graph.convs.4.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_4_lin_edge_weight'), target='core.node_processor.graph.convs.4.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_5_att'), target='core.node_processor.graph.convs.5.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_5_bias'), target='core.node_processor.graph.convs.5.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_5_lin_l_weight'), target='core.node_processor.graph.convs.5.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_5_lin_l_bias'), target='core.node_processor.graph.convs.5.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_5_lin_r_weight'), target='core.node_processor.graph.convs.5.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_5_lin_r_bias'), target='core.node_processor.graph.convs.5.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_convs_5_lin_edge_weight'), target='core.node_processor.graph.convs.5.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_norms_0_weight'), target='core.node_processor.graph.norms.0.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_norms_0_bias'), target='core.node_processor.graph.norms.0.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_norms_1_weight'), target='core.node_processor.graph.norms.1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_norms_1_bias'), target='core.node_processor.graph.norms.1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_norms_2_weight'), target='core.node_processor.graph.norms.2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_norms_2_bias'), target='core.node_processor.graph.norms.2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_norms_3_weight'), target='core.node_processor.graph.norms.3.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_norms_3_bias'), target='core.node_processor.graph.norms.3.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_norms_4_weight'), target='core.node_processor.graph.norms.4.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_node_processor_graph_norms_4_bias'), target='core.node_processor.graph.norms.4.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_self_attn_in_proj_weight'), target='core.ring_encoder.layers.0.self_attn.in_proj_weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_self_attn_in_proj_bias'), target='core.ring_encoder.layers.0.self_attn.in_proj_bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_self_attn_out_proj_weight'), target='core.ring_encoder.layers.0.self_attn.out_proj.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_self_attn_out_proj_bias'), target='core.ring_encoder.layers.0.self_attn.out_proj.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_linear1_weight'), target='core.ring_encoder.layers.0.linear1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_linear1_bias'), target='core.ring_encoder.layers.0.linear1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_linear2_weight'), target='core.ring_encoder.layers.0.linear2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_linear2_bias'), target='core.ring_encoder.layers.0.linear2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_norm1_weight'), target='core.ring_encoder.layers.0.norm1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_norm1_bias'), target='core.ring_encoder.layers.0.norm1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_norm2_weight'), target='core.ring_encoder.layers.0.norm2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_ring_encoder_layers_0_norm2_bias'), target='core.ring_encoder.layers.0.norm2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_self_attn_in_proj_weight'), target='core.mol_encoder.layers.0.self_attn.in_proj_weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_self_attn_in_proj_bias'), target='core.mol_encoder.layers.0.self_attn.in_proj_bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_self_attn_out_proj_weight'), target='core.mol_encoder.layers.0.self_attn.out_proj.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_self_attn_out_proj_bias'), target='core.mol_encoder.layers.0.self_attn.out_proj.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_linear1_weight'), target='core.mol_encoder.layers.0.linear1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_linear1_bias'), target='core.mol_encoder.layers.0.linear1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_linear2_weight'), target='core.mol_encoder.layers.0.linear2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_linear2_bias'), target='core.mol_encoder.layers.0.linear2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_norm1_weight'), target='core.mol_encoder.layers.0.norm1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_norm1_bias'), target='core.mol_encoder.layers.0.norm1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_norm2_weight'), target='core.mol_encoder.layers.0.norm2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_0_norm2_bias'), target='core.mol_encoder.layers.0.norm2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_self_attn_in_proj_weight'), target='core.mol_encoder.layers.1.self_attn.in_proj_weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_self_attn_in_proj_bias'), target='core.mol_encoder.layers.1.self_attn.in_proj_bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_self_attn_out_proj_weight'), target='core.mol_encoder.layers.1.self_attn.out_proj.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_self_attn_out_proj_bias'), target='core.mol_encoder.layers.1.self_attn.out_proj.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_linear1_weight'), target='core.mol_encoder.layers.1.linear1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_linear1_bias'), target='core.mol_encoder.layers.1.linear1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_linear2_weight'), target='core.mol_encoder.layers.1.linear2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_linear2_bias'), target='core.mol_encoder.layers.1.linear2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_norm1_weight'), target='core.mol_encoder.layers.1.norm1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_norm1_bias'), target='core.mol_encoder.layers.1.norm1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_norm2_weight'), target='core.mol_encoder.layers.1.norm2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_1_norm2_bias'), target='core.mol_encoder.layers.1.norm2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_self_attn_in_proj_weight'), target='core.mol_encoder.layers.2.self_attn.in_proj_weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_self_attn_in_proj_bias'), target='core.mol_encoder.layers.2.self_attn.in_proj_bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_self_attn_out_proj_weight'), target='core.mol_encoder.layers.2.self_attn.out_proj.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_self_attn_out_proj_bias'), target='core.mol_encoder.layers.2.self_attn.out_proj.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_linear1_weight'), target='core.mol_encoder.layers.2.linear1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_linear1_bias'), target='core.mol_encoder.layers.2.linear1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_linear2_weight'), target='core.mol_encoder.layers.2.linear2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_linear2_bias'), target='core.mol_encoder.layers.2.linear2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_norm1_weight'), target='core.mol_encoder.layers.2.norm1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_norm1_bias'), target='core.mol_encoder.layers.2.norm1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_norm2_weight'), target='core.mol_encoder.layers.2.norm2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_2_norm2_bias'), target='core.mol_encoder.layers.2.norm2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_self_attn_in_proj_weight'), target='core.mol_encoder.layers.3.self_attn.in_proj_weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_self_attn_in_proj_bias'), target='core.mol_encoder.layers.3.self_attn.in_proj_bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_self_attn_out_proj_weight'), target='core.mol_encoder.layers.3.self_attn.out_proj.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_self_attn_out_proj_bias'), target='core.mol_encoder.layers.3.self_attn.out_proj.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_linear1_weight'), target='core.mol_encoder.layers.3.linear1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_linear1_bias'), target='core.mol_encoder.layers.3.linear1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_linear2_weight'), target='core.mol_encoder.layers.3.linear2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_linear2_bias'), target='core.mol_encoder.layers.3.linear2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_norm1_weight'), target='core.mol_encoder.layers.3.norm1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_norm1_bias'), target='core.mol_encoder.layers.3.norm1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_norm2_weight'), target='core.mol_encoder.layers.3.norm2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_core_mol_encoder_layers_3_norm2_bias'), target='core.mol_encoder.layers.3.norm2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_predictors_hidden_layers_lins_0_weight'), target='predictors.hidden_layers.lins.0.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_predictors_hidden_layers_lins_0_bias'), target='predictors.hidden_layers.lins.0.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_predictors_out_layer_weight'), target='predictors.out_layer.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_predictors_out_layer_bias'), target='predictors.out_layer.bias', persistent=None), InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='b_core_node_processor_norm_running_mean'), target='core.node_processor.norm.running_mean', persistent=True), InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='b_core_node_processor_norm_running_var'), target='core.node_processor.norm.running_var', persistent=True), InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='b_core_node_processor_norm_num_batches_tracked'), target='core.node_processor.norm.num_batches_tracked', persistent=True), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='x'), target=None, persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='padded_xr'), target=None, persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='rings_mask'), target=None, persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='cbond_index'), target=None, persistent=None)], output_specs=[OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='linear_21'), target=None)])
Range constraints: {s0: VR[0, int_oo], s1: VR[0, int_oo]}

```

## ONNX model

```python
<
    ir_version=10,
    opset_imports={'pkg.onnxscript.torch_lib.common': 1, '': 18, 'pkg.onnxscript.torch_lib': 1},
    producer_name='pytorch',
    producer_version='2.6.0+cu124',
    domain=None,
    model_version=None,
>
graph(
    name=main_graph,
    inputs=(
        %"xg"<FLOAT,[s0,128]>,
        %"padded_Xr"<FLOAT,[128,64,128]>,
        %"rings_mask"<BOOL,[128,64]>,
        %"cbond_index"<INT64,[2,s1]>
    ),
    outputs=(
        %"cbond"<FLOAT,[s1,1]>
    ),
    initializers=(
        %"core.CLS"<FLOAT,[1,128]>{TorchTensor(...)},
        %"core.RING"<FLOAT,[1,128]>{TorchTensor(...)},
        %"core.END"<FLOAT,[1,128]>{TorchTensor(...)},
        %"core.node_processor.x_emb.weight"<FLOAT,[120,128]>{TorchTensor(...)},
        %"core.node_processor.lin.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.lin.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.norm.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.norm.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.0.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.0.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.0.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.0.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.0.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.0.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.0.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.1.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.1.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.1.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.1.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.1.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.1.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.1.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.2.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.2.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.2.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.2.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.2.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.2.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.3.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.3.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.3.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.3.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.3.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.3.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.3.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.4.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.4.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.4.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.4.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.4.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.4.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.4.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.5.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.5.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.5.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.5.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.5.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.5.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.convs.5.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.node_processor.graph.norms.0.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.norms.0.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.norms.1.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.norms.1.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.norms.2.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.norms.2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.norms.3.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.norms.3.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.norms.4.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.graph.norms.4.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.self_attn.in_proj_weight"<FLOAT,[384,128]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.self_attn.in_proj_bias"<FLOAT,[384]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.self_attn.out_proj.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.self_attn.out_proj.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.linear1.weight"<FLOAT,[1024,128]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.linear1.bias"<FLOAT,[1024]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.linear2.weight"<FLOAT,[128,1024]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.linear2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.norm1.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.norm1.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.norm2.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.ring_encoder.layers.0.norm2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.self_attn.in_proj_weight"<FLOAT,[384,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.self_attn.in_proj_bias"<FLOAT,[384]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.self_attn.out_proj.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.self_attn.out_proj.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.linear1.weight"<FLOAT,[1024,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.linear1.bias"<FLOAT,[1024]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.linear2.weight"<FLOAT,[128,1024]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.linear2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.norm1.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.norm1.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.norm2.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.0.norm2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.self_attn.in_proj_weight"<FLOAT,[384,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.self_attn.in_proj_bias"<FLOAT,[384]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.self_attn.out_proj.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.self_attn.out_proj.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.linear1.weight"<FLOAT,[1024,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.linear1.bias"<FLOAT,[1024]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.linear2.weight"<FLOAT,[128,1024]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.linear2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.norm1.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.norm1.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.norm2.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.1.norm2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.self_attn.in_proj_weight"<FLOAT,[384,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.self_attn.in_proj_bias"<FLOAT,[384]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.self_attn.out_proj.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.self_attn.out_proj.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.linear1.weight"<FLOAT,[1024,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.linear1.bias"<FLOAT,[1024]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.linear2.weight"<FLOAT,[128,1024]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.linear2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.norm1.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.norm1.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.norm2.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.2.norm2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.self_attn.in_proj_weight"<FLOAT,[384,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.self_attn.in_proj_bias"<FLOAT,[384]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.self_attn.out_proj.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.self_attn.out_proj.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.linear1.weight"<FLOAT,[1024,128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.linear1.bias"<FLOAT,[1024]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.linear2.weight"<FLOAT,[128,1024]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.linear2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.norm1.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.norm1.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.norm2.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"core.mol_encoder.layers.3.norm2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"predictors.hidden_layers.lins.0.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"predictors.hidden_layers.lins.0.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"predictors.out_layer.weight"<FLOAT,[1,128]>{TorchTensor(...)},
        %"predictors.out_layer.bias"<FLOAT,[1]>{TorchTensor<FLOAT,[1]>(Parameter containing: tensor([-0.0288], dtype=torch.float32, requires_grad=True), name='predictors.out_layer.bias')},
        %"core.node_processor.norm.running_mean"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.norm.running_var"<FLOAT,[128]>{TorchTensor(...)},
        %"core.node_processor.norm.num_batches_tracked"<INT64,[]>{TorchTensor<INT64,[]>(tensor(39792), name='core.node_processor.norm.num_batches_tracked')}
    ),
) {
      0 |  # node_Shape_0
           %"val_0"<?,?> ⬅️ ::Shape(%"xg") {end=1, start=0}
      1 |  # node_Squeeze_1
           %"sym_size_int_17"<INT64,[]> ⬅️ ::Squeeze(%"val_0")
      2 |  # node_Constant_2
           %"val_1"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[]>(array(0), name=None)}
      3 |  # node_Cast_3
           %"val_2"<?,?> ⬅️ ::Cast(%"val_1") {to=FLOAT}
      4 |  # node_Shape_4
           %"val_3"<?,?> ⬅️ ::Shape(%"rings_mask") {start=0}
      5 |  # node_Expand_5
           %"zeros_like"<FLOAT,[128,64]> ⬅️ ::Expand(%"val_2", %"val_3")
      6 |  # node_Constant_6
           %"val_4"<?,?> ⬅️ ::Constant() {value=Tensor<FLOAT,[]>(array(-inf, dtype=float32), name=None)}
      7 |  # node_CastLike_7
           %"val_5"<?,?> ⬅️ ::CastLike(%"val_4", %"zeros_like")
      8 |  # node_Where_8
           %"masked_fill"<FLOAT,[128,64]> ⬅️ ::Where(%"rings_mask", %"val_5", %"zeros_like")
      9 |  # node_Transpose_9
           %"transpose"<FLOAT,[64,128,128]> ⬅️ ::Transpose(%"padded_Xr") {perm=[1, 0, 2]}
     10 |  # node_Transpose_10
           %"val_6"<?,?> ⬅️ ::Transpose(%"core.ring_encoder.layers.0.self_attn.in_proj_weight"{...}) {perm=[1, 0]}
     11 |  # node_MatMul_11
           %"val_7"<?,?> ⬅️ ::MatMul(%"transpose", %"val_6")
     12 |  # node_Add_12
           %"linear"<FLOAT,[64,128,384]> ⬅️ ::Add(%"val_7", %"core.ring_encoder.layers.0.self_attn.in_proj_bias"{...})
     13 |  # node_Constant_13
           %"val_8"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[4]>(array([ 64, 128,   3, 128]), name=None)}
     14 |  # node_Cast_14
           %"val_9"<?,?> ⬅️ ::Cast(%"val_8") {to=INT64}
     15 |  # node_Reshape_15
           %"view"<FLOAT,[64,128,3,128]> ⬅️ ::Reshape(%"linear", %"val_9") {allowzero=True}
     16 |  # node_Constant_16
           %"val_10"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([0]), name=None)}
     17 |  # node_Unsqueeze_17
           %"unsqueeze"<FLOAT,[1,64,128,3,128]> ⬅️ ::Unsqueeze(%"view", %"val_10")
     18 |  # node_Transpose_18
           %"transpose_1"<FLOAT,[3,64,128,1,128]> ⬅️ ::Transpose(%"unsqueeze") {perm=[3, 1, 2, 0, 4]}
     19 |  # node_Constant_19
           %"val_11"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-2]), name=None)}
     20 |  # node_Squeeze_20
           %"squeeze"<FLOAT,[3,64,128,128]> ⬅️ ::Squeeze(%"transpose_1", %"val_11")
     21 |  # node_Identity_21
           %"clone"<FLOAT,[3,64,128,128]> ⬅️ ::Identity(%"squeeze")
     22 |  # node_Gather_22
           %"select"<FLOAT,[64,128,128]> ⬅️ ::Gather(%"clone", %"val_1") {axis=0}
     23 |  # node_Constant_23
           %"val_12"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[]>(array(1), name=None)}
     24 |  # node_Gather_24
           %"select_1"<FLOAT,[64,128,128]> ⬅️ ::Gather(%"clone", %"val_12") {axis=0}
     25 |  # node_Constant_25
           %"val_13"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[]>(array(2), name=None)}
     26 |  # node_Gather_26
           %"select_2"<FLOAT,[64,128,128]> ⬅️ ::Gather(%"clone", %"val_13") {axis=0}
     27 |  # node_Constant_27
           %"val_14"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[3]>(array([ 64, 256,  64]), name=None)}
     28 |  # node_Cast_28
           %"val_15"<?,?> ⬅️ ::Cast(%"val_14") {to=INT64}
     29 |  # node_Reshape_29
           %"view_1"<FLOAT,[64,256,64]> ⬅️ ::Reshape(%"select", %"val_15") {allowzero=True}
     30 |  # node_Transpose_30
           %"transpose_2"<FLOAT,[256,64,64]> ⬅️ ::Transpose(%"view_1") {perm=[1, 0, 2]}
     31 |  # node_Cast_31
           %"val_16"<?,?> ⬅️ ::Cast(%"val_14") {to=INT64}
     32 |  # node_Reshape_32
           %"view_2"<FLOAT,[64,256,64]> ⬅️ ::Reshape(%"select_1", %"val_16") {allowzero=True}
     33 |  # node_Transpose_33
           %"transpose_3"<FLOAT,[256,64,64]> ⬅️ ::Transpose(%"view_2") {perm=[1, 0, 2]}
     34 |  # node_Cast_34
           %"val_17"<?,?> ⬅️ ::Cast(%"val_14") {to=INT64}
     35 |  # node_Reshape_35
           %"view_3"<FLOAT,[64,256,64]> ⬅️ ::Reshape(%"select_2", %"val_17") {allowzero=True}
     36 |  # node_Transpose_36
           %"transpose_4"<FLOAT,[256,64,64]> ⬅️ ::Transpose(%"view_3") {perm=[1, 0, 2]}
     37 |  # node_Constant_37
           %"val_18"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[4]>(array([128,   1,   1,  64]), name=None)}
     38 |  # node_Cast_38
           %"val_19"<?,?> ⬅️ ::Cast(%"val_18") {to=INT64}
     39 |  # node_Reshape_39
           %"view_5"<FLOAT,[128,1,1,64]> ⬅️ ::Reshape(%"masked_fill", %"val_19") {allowzero=True}
     40 |  # node_Constant_40
           %"val_20"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[4]>(array([-1,  2, -1, -1]), name=None)}
     41 |  # node_Cast_41
           %"val_21"<?,?> ⬅️ ::Cast(%"val_20") {to=INT64}
     42 |  # node_Abs_42
           %"val_22"<?,?> ⬅️ ::Abs(%"val_21")
     43 |  # node_Expand_43
           %"expand_1"<FLOAT,[128,2,1,64]> ⬅️ ::Expand(%"view_5", %"val_22")
     44 |  # node_Identity_44
           %"clone_1"<FLOAT,[128,2,1,64]> ⬅️ ::Identity(%"expand_1")
     45 |  # node_Constant_45
           %"val_23"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[3]>(array([256,   1,  64]), name=None)}
     46 |  # node_Cast_46
           %"val_24"<?,?> ⬅️ ::Cast(%"val_23") {to=INT64}
     47 |  # node_Reshape_47
           %"_unsafe_view"<FLOAT,[256,1,64]> ⬅️ ::Reshape(%"clone_1", %"val_24") {allowzero=True}
     48 |  # node_Constant_48
           %"val_25"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[4]>(array([128,   2,  -1,  64]), name=None)}
     49 |  # node_Cast_49
           %"val_26"<?,?> ⬅️ ::Cast(%"val_25") {to=INT64}
     50 |  # node_Reshape_50
           %"view_6"<FLOAT,[128,2,1,64]> ⬅️ ::Reshape(%"_unsafe_view", %"val_26") {allowzero=True}
     51 |  # node_Constant_51
           %"val_27"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[4]>(array([128,   2,  64,  64]), name=None)}
     52 |  # node_Cast_52
           %"val_28"<?,?> ⬅️ ::Cast(%"val_27") {to=INT64}
     53 |  # node_Reshape_53
           %"view_7"<FLOAT,[128,2,64,64]> ⬅️ ::Reshape(%"transpose_2", %"val_28") {allowzero=True}
     54 |  # node_Cast_54
           %"val_29"<?,?> ⬅️ ::Cast(%"val_27") {to=INT64}
     55 |  # node_Reshape_55
           %"view_8"<FLOAT,[128,2,64,64]> ⬅️ ::Reshape(%"transpose_3", %"val_29") {allowzero=True}
     56 |  # node_Cast_56
           %"val_30"<?,?> ⬅️ ::Cast(%"val_27") {to=INT64}
     57 |  # node_Reshape_57
           %"view_9"<FLOAT,[128,2,64,64]> ⬅️ ::Reshape(%"transpose_4", %"val_30") {allowzero=True}
     58 |  # node_Shape_58
           %"val_31"<?,?> ⬅️ ::Shape(%"view_7") {start=0}
     59 |  # node_Constant_59
           %"val_32"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
     60 |  # node_Gather_60
           %"val_33"<?,?> ⬅️ ::Gather(%"val_31", %"val_32") {axis=0}
     61 |  # node_CastLike_61
           %"val_34"<?,?> ⬅️ ::CastLike(%"val_33", %"view_7")
     62 |  # node_Constant_62
           %"val_35"<?,?> ⬅️ ::Constant() {value_float=1.0}
     63 |  # node_CastLike_63
           %"val_36"<?,?> ⬅️ ::CastLike(%"val_35", %"view_7")
     64 |  # node_Sqrt_64
           %"val_37"<?,?> ⬅️ ::Sqrt(%"val_34")
     65 |  # node_Div_65
           %"val_38"<?,?> ⬅️ ::Div(%"val_36", %"val_37")
     66 |  # node_CastLike_66
           %"val_39"<?,?> ⬅️ ::CastLike(%"val_38", %"view_7")
     67 |  # node_Shape_67
           %"val_40"<?,?> ⬅️ ::Shape(%"view_8") {start=0}
     68 |  # node_Constant_68
           %"val_41"<?,?> ⬅️ ::Constant() {value_ints=[9223372036854775807]}
     69 |  # node_Constant_69
           %"val_42"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
     70 |  # node_Slice_70
           %"val_43"<?,?> ⬅️ ::Slice(%"val_40", %"val_42", %"val_41")
     71 |  # node_Slice_71
           %"val_44"<?,?> ⬅️ ::Slice(%"val_40", %"val_11", %"val_42")
     72 |  # node_Constant_72
           %"val_45"<?,?> ⬅️ ::Constant() {value_ints=[-9223372036854775808]}
     73 |  # node_Slice_73
           %"val_46"<?,?> ⬅️ ::Slice(%"val_40", %"val_45", %"val_11")
     74 |  # node_Constant_74
           %"val_47"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
     75 |  # node_Concat_75
           %"val_48"<?,?> ⬅️ ::Concat(%"val_47", %"val_44", %"val_43") {axis=0}
     76 |  # node_Reshape_76
           %"val_49"<?,?> ⬅️ ::Reshape(%"view_8", %"val_48") {allowzero=0}
     77 |  # node_Transpose_77
           %"val_50"<?,?> ⬅️ ::Transpose(%"val_49") {perm=[0, 2, 1]}
     78 |  # node_Concat_78
           %"val_51"<?,?> ⬅️ ::Concat(%"val_46", %"val_43", %"val_44") {axis=0}
     79 |  # node_Reshape_79
           %"val_52"<?,?> ⬅️ ::Reshape(%"val_50", %"val_51") {allowzero=0}
     80 |  # node_Sqrt_80
           %"val_53"<?,?> ⬅️ ::Sqrt(%"val_39")
     81 |  # node_Mul_81
           %"val_54"<?,?> ⬅️ ::Mul(%"view_7", %"val_53")
     82 |  # node_Sqrt_82
           %"val_55"<?,?> ⬅️ ::Sqrt(%"val_39")
     83 |  # node_Mul_83
           %"val_56"<?,?> ⬅️ ::Mul(%"val_52", %"val_55")
     84 |  # node_MatMul_84
           %"val_57"<?,?> ⬅️ ::MatMul(%"val_54", %"val_56")
     85 |  # node_Add_85
           %"val_58"<?,?> ⬅️ ::Add(%"val_57", %"view_6")
     86 |  # node_Softmax_86
           %"val_59"<?,?> ⬅️ ::Softmax(%"val_58") {axis=-1}
     87 |  # node_MatMul_87
           %"scaled_dot_product_attention"<FLOAT,[128,2,64,64]> ⬅️ ::MatMul(%"val_59", %"view_9")
     88 |  # node_Transpose_88
           %"permute"<FLOAT,[64,128,2,64]> ⬅️ ::Transpose(%"scaled_dot_product_attention") {perm=[2, 0, 1, 3]}
     89 |  # node_Constant_89
           %"val_60"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[2]>(array([8192,  128]), name=None)}
     90 |  # node_Cast_90
           %"val_61"<?,?> ⬅️ ::Cast(%"val_60") {to=INT64}
     91 |  # node_Reshape_91
           %"view_10"<FLOAT,[8192,128]> ⬅️ ::Reshape(%"permute", %"val_61") {allowzero=True}
     92 |  # node_Gemm_92
           %"linear_1"<FLOAT,[8192,128]> ⬅️ ::Gemm(%"view_10", %"core.ring_encoder.layers.0.self_attn.out_proj.weight"{...}, %"core.ring_encoder.layers.0.self_attn.out_proj.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
     93 |  # node_Constant_93
           %"val_62"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[3]>(array([ 64, 128, 128]), name=None)}
     94 |  # node_Cast_94
           %"val_63"<?,?> ⬅️ ::Cast(%"val_62") {to=INT64}
     95 |  # node_Reshape_95
           %"view_11"<FLOAT,[64,128,128]> ⬅️ ::Reshape(%"linear_1", %"val_63") {allowzero=True}
     96 |  # node_Transpose_96
           %"transpose_5"<FLOAT,[128,64,128]> ⬅️ ::Transpose(%"view_11") {perm=[1, 0, 2]}
     97 |  # node_Identity_97
           %"clone_2"<FLOAT,[128,64,128]> ⬅️ ::Identity(%"transpose_5")
     98 |  # node_Add_98
           %"add"<FLOAT,[128,64,128]> ⬅️ ::Add(%"padded_Xr", %"clone_2")
     99 |  # node_LayerNormalization_99
           %"layer_norm"<FLOAT,[128,64,128]>, %"val_64"<?,?>, %"val_65"<?,?> ⬅️ ::LayerNormalization(%"add", %"core.ring_encoder.layers.0.norm1.weight"{...}, %"core.ring_encoder.layers.0.norm1.bias"{...}) {stash_type=1, epsilon=1e-05, axis=-1}
    100 |  # node_Transpose_100
           %"val_66"<?,?> ⬅️ ::Transpose(%"core.ring_encoder.layers.0.linear1.weight"{...}) {perm=[1, 0]}
    101 |  # node_MatMul_101
           %"val_67"<?,?> ⬅️ ::MatMul(%"layer_norm", %"val_66")
    102 |  # node_Add_102
           %"linear_2"<FLOAT,[128,64,1024]> ⬅️ ::Add(%"val_67", %"core.ring_encoder.layers.0.linear1.bias"{...})
    103 |  # node_Relu_103
           %"relu"<FLOAT,[128,64,1024]> ⬅️ ::Relu(%"linear_2")
    104 |  # node_Identity_104
           %"clone_3"<FLOAT,[128,64,1024]> ⬅️ ::Identity(%"relu")
    105 |  # node_Transpose_105
           %"val_68"<?,?> ⬅️ ::Transpose(%"core.ring_encoder.layers.0.linear2.weight"{...}) {perm=[1, 0]}
    106 |  # node_MatMul_106
           %"val_69"<?,?> ⬅️ ::MatMul(%"clone_3", %"val_68")
    107 |  # node_Add_107
           %"linear_3"<FLOAT,[128,64,128]> ⬅️ ::Add(%"val_69", %"core.ring_encoder.layers.0.linear2.bias"{...})
    108 |  # node_Identity_108
           %"clone_4"<FLOAT,[128,64,128]> ⬅️ ::Identity(%"linear_3")
    109 |  # node_Add_109
           %"add_1"<FLOAT,[128,64,128]> ⬅️ ::Add(%"layer_norm", %"clone_4")
    110 |  # node_LayerNormalization_110
           %"layer_norm_1"<FLOAT,[128,64,128]>, %"val_70"<?,?>, %"val_71"<?,?> ⬅️ ::LayerNormalization(%"add_1", %"core.ring_encoder.layers.0.norm2.weight"{...}, %"core.ring_encoder.layers.0.norm2.bias"{...}) {stash_type=1, epsilon=1e-05, axis=-1}
    111 |  # node_Unsqueeze_111
           %"unsqueeze_1"<BOOL,[128,64,1]> ⬅️ ::Unsqueeze(%"rings_mask", %"val_42")
    112 |  # node_Constant_112
           %"val_72"<?,?> ⬅️ ::Constant() {value=Tensor<FLOAT,[]>(array(0., dtype=float32), name=None)}
    113 |  # node_CastLike_113
           %"val_73"<?,?> ⬅️ ::CastLike(%"val_72", %"layer_norm_1")
    114 |  # node_Where_114
           %"masked_fill_1"<FLOAT,[128,64,128]> ⬅️ ::Where(%"unsqueeze_1", %"val_73", %"layer_norm_1")
    115 |  # node_Abs_115
           %"abs_1"<FLOAT,[128,64,128]> ⬅️ ::Abs(%"masked_fill_1")
    116 |  # node_ArgMax_116
           %"argmax"<INT64,[128,128]> ⬅️ ::ArgMax(%"abs_1") {select_last_index=0, keepdims=False, axis=-2}
    117 |  # node_Unsqueeze_117
           %"unsqueeze_2"<INT64,[128,1,128]> ⬅️ ::Unsqueeze(%"argmax", %"val_11")
    118 |  # node_Cast_118
           %"val_74"<?,?> ⬅️ ::Cast(%"unsqueeze_2") {to=INT64}
    119 |  # node_GatherElements_119
           %"gather"<FLOAT,[128,1,128]> ⬅️ ::GatherElements(%"masked_fill_1", %"val_74") {axis=-2}
    120 |  # node_Squeeze_120
           %"squeeze_1"<FLOAT,[128,128]> ⬅️ ::Squeeze(%"gather", %"val_11")
    121 |  # node_Cast_121
           %"val_75"<?,?> ⬅️ ::Cast(%"rings_mask") {to=BOOL}
    122 |  # node_Cast_122
           %"val_76"<?,?> ⬅️ ::Cast(%"val_75") {to=INT64}
    123 |  # node_Constant_123
           %"val_77"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    124 |  # node_Reshape_124
           %"val_78"<?,?> ⬅️ ::Reshape(%"val_12", %"val_77") {allowzero=0}
    125 |  # node_ReduceMin_125
           %"val_79"<?,?> ⬅️ ::ReduceMin(%"val_76", %"val_78") {noop_with_empty_axes=0, keepdims=False}
    126 |  # node_Cast_126
           %"all_1"<BOOL,[128]> ⬅️ ::Cast(%"val_79") {to=BOOL}
    127 |  # node_Concat_127
           %"cat"<FLOAT,[s0 + 131,128]> ⬅️ ::Concat(%"core.CLS"{...}, %"xg", %"core.RING"{...}, %"squeeze_1", %"core.END"{...}) {axis=0}
    128 |  # node_Constant_128
           %"val_80"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[]>(array(131), name=None)}
    129 |  # node_Add_129
           %"add_613"<INT64,[]> ⬅️ ::Add(%"val_80", %"sym_size_int_17")
    130 |  # node_Unsqueeze_130
           %"unsqueeze_3"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Unsqueeze(%"cat", %"val_10")
    131 |  # node_Add_131
           %"add_8"<INT64,[]> ⬅️ ::Add(%"val_13", %"sym_size_int_17")
    132 |  # node_Constant_132
           %"val_81"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    133 |  # node_Reshape_133
           %"val_82"<?,?> ⬅️ ::Reshape(%"add_8", %"val_81") {allowzero=0}
    134 |  # node_Concat_134
           %"val_83"<?,?> ⬅️ ::Concat(%"val_82") {axis=0}
    135 |  # node_Cast_135
           %"val_84"<?,?> ⬅️ ::Cast(%"val_83") {to=INT64}
    136 |  # node_Constant_136
           %"val_85"<?,?> ⬅️ ::Constant() {value_float=0.0}
    137 |  # node_Cast_137
           %"val_86"<?,?> ⬅️ ::Cast(%"val_85") {to=BOOL}
    138 |  # node_Expand_138
           %"zeros"<BOOL,[s0 + 2]> ⬅️ ::Expand(%"val_86", %"val_84")
    139 |  # node_Constant_139
           %"val_87"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([1]), name=None)}
    140 |  # node_Cast_140
           %"val_88"<?,?> ⬅️ ::Cast(%"val_87") {to=INT64}
    141 |  # node_Constant_141
           %"val_89"<?,?> ⬅️ ::Constant() {value_float=0.0}
    142 |  # node_Cast_142
           %"val_90"<?,?> ⬅️ ::Cast(%"val_89") {to=BOOL}
    143 |  # node_Expand_143
           %"zeros_1"<BOOL,[1]> ⬅️ ::Expand(%"val_90", %"val_88")
    144 |  # node_Concat_144
           %"cat_1"<BOOL,[s0 + 131]> ⬅️ ::Concat(%"zeros", %"all_1", %"zeros_1") {axis=0}
    145 |  # node_Unsqueeze_145
           %"unsqueeze_4"<BOOL,[1,s0 + 131]> ⬅️ ::Unsqueeze(%"cat_1", %"val_10")
    146 |  # node_Cast_146
           %"val_91"<?,?> ⬅️ ::Cast(%"val_1") {to=FLOAT}
    147 |  # node_Shape_147
           %"val_92"<?,?> ⬅️ ::Shape(%"unsqueeze_4") {start=0}
    148 |  # node_Expand_148
           %"zeros_like_1"<FLOAT,[1,s0 + 131]> ⬅️ ::Expand(%"val_91", %"val_92")
    149 |  # node_CastLike_149
           %"val_93"<?,?> ⬅️ ::CastLike(%"val_4", %"zeros_like_1")
    150 |  # node_Where_150
           %"masked_fill_2"<FLOAT,[1,s0 + 131]> ⬅️ ::Where(%"unsqueeze_4", %"val_93", %"zeros_like_1")
    151 |  # node_Transpose_151
           %"transpose_6"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Transpose(%"unsqueeze_3") {perm=[1, 0, 2]}
    152 |  # node_Transpose_152
           %"val_94"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.0.self_attn.in_proj_weight"{...}) {perm=[1, 0]}
    153 |  # node_MatMul_153
           %"val_95"<?,?> ⬅️ ::MatMul(%"transpose_6", %"val_94")
    154 |  # node_Add_154
           %"linear_4"<FLOAT,[s0 + 131,1,384]> ⬅️ ::Add(%"val_95", %"core.mol_encoder.layers.0.self_attn.in_proj_bias"{...})
    155 |  # node_Constant_155
           %"val_96"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    156 |  # node_Reshape_156
           %"val_97"<?,?> ⬅️ ::Reshape(%"add_613", %"val_96") {allowzero=0}
    157 |  # node_Constant_157
           %"val_98"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([3]), name=None)}
    158 |  # node_Constant_158
           %"val_99"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([128]), name=None)}
    159 |  # node_Concat_159
           %"val_100"<?,?> ⬅️ ::Concat(%"val_97", %"val_87", %"val_98", %"val_99") {axis=0}
    160 |  # node_Cast_160
           %"val_101"<?,?> ⬅️ ::Cast(%"val_100") {to=INT64}
    161 |  # node_Reshape_161
           %"view_12"<FLOAT,[s0 + 131,1,3,128]> ⬅️ ::Reshape(%"linear_4", %"val_101") {allowzero=True}
    162 |  # node_Unsqueeze_162
           %"unsqueeze_5"<FLOAT,[1,s0 + 131,1,3,128]> ⬅️ ::Unsqueeze(%"view_12", %"val_10")
    163 |  # node_Transpose_163
           %"transpose_7"<FLOAT,[3,s0 + 131,1,1,128]> ⬅️ ::Transpose(%"unsqueeze_5") {perm=[3, 1, 2, 0, 4]}
    164 |  # node_Squeeze_164
           %"squeeze_2"<FLOAT,[3,s0 + 131,1,128]> ⬅️ ::Squeeze(%"transpose_7", %"val_11")
    165 |  # node_Identity_165
           %"clone_5"<FLOAT,[3,s0 + 131,1,128]> ⬅️ ::Identity(%"squeeze_2")
    166 |  # node_Gather_166
           %"select_3"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_5", %"val_1") {axis=0}
    167 |  # node_Gather_167
           %"select_4"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_5", %"val_12") {axis=0}
    168 |  # node_Gather_168
           %"select_5"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_5", %"val_13") {axis=0}
    169 |  # node_Constant_169
           %"val_102"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    170 |  # node_Reshape_170
           %"val_103"<?,?> ⬅️ ::Reshape(%"add_613", %"val_102") {allowzero=0}
    171 |  # node_Constant_171
           %"val_104"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([4]), name=None)}
    172 |  # node_Constant_172
           %"val_105"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([32]), name=None)}
    173 |  # node_Concat_173
           %"val_106"<?,?> ⬅️ ::Concat(%"val_103", %"val_104", %"val_105") {axis=0}
    174 |  # node_Cast_174
           %"val_107"<?,?> ⬅️ ::Cast(%"val_106") {to=INT64}
    175 |  # node_Reshape_175
           %"view_13"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_3", %"val_107") {allowzero=True}
    176 |  # node_Transpose_176
           %"transpose_8"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_13") {perm=[1, 0, 2]}
    177 |  # node_Constant_177
           %"val_108"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    178 |  # node_Reshape_178
           %"val_109"<?,?> ⬅️ ::Reshape(%"add_613", %"val_108") {allowzero=0}
    179 |  # node_Concat_179
           %"val_110"<?,?> ⬅️ ::Concat(%"val_109", %"val_104", %"val_105") {axis=0}
    180 |  # node_Cast_180
           %"val_111"<?,?> ⬅️ ::Cast(%"val_110") {to=INT64}
    181 |  # node_Reshape_181
           %"view_14"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_4", %"val_111") {allowzero=True}
    182 |  # node_Transpose_182
           %"transpose_9"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_14") {perm=[1, 0, 2]}
    183 |  # node_Constant_183
           %"val_112"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    184 |  # node_Reshape_184
           %"val_113"<?,?> ⬅️ ::Reshape(%"add_613", %"val_112") {allowzero=0}
    185 |  # node_Concat_185
           %"val_114"<?,?> ⬅️ ::Concat(%"val_113", %"val_104", %"val_105") {axis=0}
    186 |  # node_Cast_186
           %"val_115"<?,?> ⬅️ ::Cast(%"val_114") {to=INT64}
    187 |  # node_Reshape_187
           %"view_15"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_5", %"val_115") {allowzero=True}
    188 |  # node_Transpose_188
           %"transpose_10"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_15") {perm=[1, 0, 2]}
    189 |  # node_Constant_189
           %"val_116"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    190 |  # node_Reshape_190
           %"val_117"<?,?> ⬅️ ::Reshape(%"add_613", %"val_116") {allowzero=0}
    191 |  # node_Concat_191
           %"val_118"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_117", %"val_105") {axis=0}
    192 |  # node_Cast_192
           %"val_119"<?,?> ⬅️ ::Cast(%"val_118") {to=INT64}
    193 |  # node_Reshape_193
           %"view_19"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_8", %"val_119") {allowzero=True}
    194 |  # node_Constant_194
           %"val_120"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    195 |  # node_Reshape_195
           %"val_121"<?,?> ⬅️ ::Reshape(%"add_613", %"val_120") {allowzero=0}
    196 |  # node_Concat_196
           %"val_122"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_121", %"val_105") {axis=0}
    197 |  # node_Cast_197
           %"val_123"<?,?> ⬅️ ::Cast(%"val_122") {to=INT64}
    198 |  # node_Reshape_198
           %"view_20"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_9", %"val_123") {allowzero=True}
    199 |  # node_Constant_199
           %"val_124"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    200 |  # node_Reshape_200
           %"val_125"<?,?> ⬅️ ::Reshape(%"add_613", %"val_124") {allowzero=0}
    201 |  # node_Concat_201
           %"val_126"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_125", %"val_105") {axis=0}
    202 |  # node_Cast_202
           %"val_127"<?,?> ⬅️ ::Cast(%"val_126") {to=INT64}
    203 |  # node_Reshape_203
           %"view_21"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_10", %"val_127") {allowzero=True}
    204 |  # node_Constant_204
           %"val_128"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    205 |  # node_Reshape_205
           %"val_129"<?,?> ⬅️ ::Reshape(%"add_613", %"val_128") {allowzero=0}
    206 |  # node_Concat_206
           %"val_130"<?,?> ⬅️ ::Concat(%"val_87", %"val_87", %"val_87", %"val_129") {axis=0}
    207 |  # node_Cast_207
           %"val_131"<?,?> ⬅️ ::Cast(%"val_130") {to=INT64}
    208 |  # node_Reshape_208
           %"view_22"<FLOAT,[1,1,1,s0 + 131]> ⬅️ ::Reshape(%"masked_fill_2", %"val_131") {allowzero=True}
    209 |  # node_Constant_209
           %"val_132"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[4]>(array([-1,  4, -1, -1]), name=None)}
    210 |  # node_Cast_210
           %"val_133"<?,?> ⬅️ ::Cast(%"val_132") {to=INT64}
    211 |  # node_Abs_211
           %"val_134"<?,?> ⬅️ ::Abs(%"val_133")
    212 |  # node_Expand_212
           %"expand_3"<FLOAT,[1,4,1,s0 + 131]> ⬅️ ::Expand(%"view_22", %"val_134")
    213 |  # node_Constant_213
           %"val_135"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    214 |  # node_Reshape_214
           %"val_136"<?,?> ⬅️ ::Reshape(%"add_613", %"val_135") {allowzero=0}
    215 |  # node_Concat_215
           %"val_137"<?,?> ⬅️ ::Concat(%"val_104", %"val_87", %"val_136") {axis=0}
    216 |  # node_Cast_216
           %"val_138"<?,?> ⬅️ ::Cast(%"val_137") {to=INT64}
    217 |  # node_Reshape_217
           %"view_23"<FLOAT,[4,1,s0 + 131]> ⬅️ ::Reshape(%"expand_3", %"val_138") {allowzero=True}
    218 |  # node_Constant_218
           %"val_139"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    219 |  # node_Reshape_219
           %"val_140"<?,?> ⬅️ ::Reshape(%"add_613", %"val_139") {allowzero=0}
    220 |  # node_Concat_220
           %"val_141"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_42", %"val_140") {axis=0}
    221 |  # node_Cast_221
           %"val_142"<?,?> ⬅️ ::Cast(%"val_141") {to=INT64}
    222 |  # node_Reshape_222
           %"view_24"<FLOAT,[1,4,1,s0 + 131]> ⬅️ ::Reshape(%"view_23", %"val_142") {allowzero=True}
    223 |  # node_Shape_223
           %"val_143"<?,?> ⬅️ ::Shape(%"view_19") {start=0}
    224 |  # node_Constant_224
           %"val_144"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    225 |  # node_Gather_225
           %"val_145"<?,?> ⬅️ ::Gather(%"val_143", %"val_144") {axis=0}
    226 |  # node_CastLike_226
           %"val_146"<?,?> ⬅️ ::CastLike(%"val_145", %"view_19")
    227 |  # node_Constant_227
           %"val_147"<?,?> ⬅️ ::Constant() {value_float=1.0}
    228 |  # node_CastLike_228
           %"val_148"<?,?> ⬅️ ::CastLike(%"val_147", %"view_19")
    229 |  # node_Sqrt_229
           %"val_149"<?,?> ⬅️ ::Sqrt(%"val_146")
    230 |  # node_Div_230
           %"val_150"<?,?> ⬅️ ::Div(%"val_148", %"val_149")
    231 |  # node_CastLike_231
           %"val_151"<?,?> ⬅️ ::CastLike(%"val_150", %"view_19")
    232 |  # node_Shape_232
           %"val_152"<?,?> ⬅️ ::Shape(%"view_20") {start=0}
    233 |  # node_Constant_233
           %"val_153"<?,?> ⬅️ ::Constant() {value_ints=[9223372036854775807]}
    234 |  # node_Slice_234
           %"val_154"<?,?> ⬅️ ::Slice(%"val_152", %"val_42", %"val_153")
    235 |  # node_Slice_235
           %"val_155"<?,?> ⬅️ ::Slice(%"val_152", %"val_11", %"val_42")
    236 |  # node_Constant_236
           %"val_156"<?,?> ⬅️ ::Constant() {value_ints=[-9223372036854775808]}
    237 |  # node_Slice_237
           %"val_157"<?,?> ⬅️ ::Slice(%"val_152", %"val_156", %"val_11")
    238 |  # node_Constant_238
           %"val_158"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    239 |  # node_Concat_239
           %"val_159"<?,?> ⬅️ ::Concat(%"val_158", %"val_155", %"val_154") {axis=0}
    240 |  # node_Reshape_240
           %"val_160"<?,?> ⬅️ ::Reshape(%"view_20", %"val_159") {allowzero=0}
    241 |  # node_Transpose_241
           %"val_161"<?,?> ⬅️ ::Transpose(%"val_160") {perm=[0, 2, 1]}
    242 |  # node_Concat_242
           %"val_162"<?,?> ⬅️ ::Concat(%"val_157", %"val_154", %"val_155") {axis=0}
    243 |  # node_Reshape_243
           %"val_163"<?,?> ⬅️ ::Reshape(%"val_161", %"val_162") {allowzero=0}
    244 |  # node_Sqrt_244
           %"val_164"<?,?> ⬅️ ::Sqrt(%"val_151")
    245 |  # node_Mul_245
           %"val_165"<?,?> ⬅️ ::Mul(%"view_19", %"val_164")
    246 |  # node_Sqrt_246
           %"val_166"<?,?> ⬅️ ::Sqrt(%"val_151")
    247 |  # node_Mul_247
           %"val_167"<?,?> ⬅️ ::Mul(%"val_163", %"val_166")
    248 |  # node_MatMul_248
           %"val_168"<?,?> ⬅️ ::MatMul(%"val_165", %"val_167")
    249 |  # node_Add_249
           %"val_169"<?,?> ⬅️ ::Add(%"val_168", %"view_24")
    250 |  # node_Softmax_250
           %"val_170"<?,?> ⬅️ ::Softmax(%"val_169") {axis=-1}
    251 |  # node_MatMul_251
           %"scaled_dot_product_attention_1"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::MatMul(%"val_170", %"view_21")
    252 |  # node_Transpose_252
           %"permute_1"<FLOAT,[s0 + 131,1,4,32]> ⬅️ ::Transpose(%"scaled_dot_product_attention_1") {perm=[2, 0, 1, 3]}
    253 |  # node_Constant_253
           %"val_171"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    254 |  # node_Reshape_254
           %"val_172"<?,?> ⬅️ ::Reshape(%"add_613", %"val_171") {allowzero=0}
    255 |  # node_Concat_255
           %"val_173"<?,?> ⬅️ ::Concat(%"val_172", %"val_99") {axis=0}
    256 |  # node_Cast_256
           %"val_174"<?,?> ⬅️ ::Cast(%"val_173") {to=INT64}
    257 |  # node_Reshape_257
           %"view_25"<FLOAT,[s0 + 131,128]> ⬅️ ::Reshape(%"permute_1", %"val_174") {allowzero=True}
    258 |  # node_Gemm_258
           %"linear_5"<FLOAT,[s0 + 131,128]> ⬅️ ::Gemm(%"view_25", %"core.mol_encoder.layers.0.self_attn.out_proj.weight"{...}, %"core.mol_encoder.layers.0.self_attn.out_proj.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    259 |  # node_Constant_259
           %"val_175"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    260 |  # node_Reshape_260
           %"val_176"<?,?> ⬅️ ::Reshape(%"add_613", %"val_175") {allowzero=0}
    261 |  # node_Concat_261
           %"val_177"<?,?> ⬅️ ::Concat(%"val_176", %"val_87", %"val_99") {axis=0}
    262 |  # node_Cast_262
           %"val_178"<?,?> ⬅️ ::Cast(%"val_177") {to=INT64}
    263 |  # node_Reshape_263
           %"view_26"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Reshape(%"linear_5", %"val_178") {allowzero=True}
    264 |  # node_Transpose_264
           %"transpose_11"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Transpose(%"view_26") {perm=[1, 0, 2]}
    265 |  # node_Identity_265
           %"clone_6"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Identity(%"transpose_11")
    266 |  # node_Add_266
           %"add_130"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"unsqueeze_3", %"clone_6")
    267 |  # node_LayerNormalization_267
           %"layer_norm_2"<FLOAT,[1,s0 + 131,128]>, %"val_179"<?,?>, %"val_180"<?,?> ⬅️ ::LayerNormalization(%"add_130", %"core.mol_encoder.layers.0.norm1.weight"{...}, %"core.mol_encoder.layers.0.norm1.bias"{...}) {stash_type=1, epsilon=1e-05, axis=-1}
    268 |  # node_Transpose_268
           %"val_181"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.0.linear1.weight"{...}) {perm=[1, 0]}
    269 |  # node_MatMul_269
           %"val_182"<?,?> ⬅️ ::MatMul(%"layer_norm_2", %"val_181")
    270 |  # node_Add_270
           %"linear_6"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Add(%"val_182", %"core.mol_encoder.layers.0.linear1.bias"{...})
    271 |  # node_Relu_271
           %"relu_1"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Relu(%"linear_6")
    272 |  # node_Identity_272
           %"clone_7"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Identity(%"relu_1")
    273 |  # node_Transpose_273
           %"val_183"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.0.linear2.weight"{...}) {perm=[1, 0]}
    274 |  # node_MatMul_274
           %"val_184"<?,?> ⬅️ ::MatMul(%"clone_7", %"val_183")
    275 |  # node_Add_275
           %"linear_7"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"val_184", %"core.mol_encoder.layers.0.linear2.bias"{...})
    276 |  # node_Identity_276
           %"clone_8"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Identity(%"linear_7")
    277 |  # node_Add_277
           %"add_152"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"layer_norm_2", %"clone_8")
    278 |  # node_LayerNormalization_278
           %"layer_norm_3"<FLOAT,[1,s0 + 131,128]>, %"val_185"<?,?>, %"val_186"<?,?> ⬅️ ::LayerNormalization(%"add_152", %"core.mol_encoder.layers.0.norm2.weight"{...}, %"core.mol_encoder.layers.0.norm2.bias"{...}) {stash_type=1, epsilon=1e-05, axis=-1}
    279 |  # node_Transpose_279
           %"transpose_12"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Transpose(%"layer_norm_3") {perm=[1, 0, 2]}
    280 |  # node_Transpose_280
           %"val_187"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.1.self_attn.in_proj_weight"{...}) {perm=[1, 0]}
    281 |  # node_MatMul_281
           %"val_188"<?,?> ⬅️ ::MatMul(%"transpose_12", %"val_187")
    282 |  # node_Add_282
           %"linear_8"<FLOAT,[s0 + 131,1,384]> ⬅️ ::Add(%"val_188", %"core.mol_encoder.layers.1.self_attn.in_proj_bias"{...})
    283 |  # node_Constant_283
           %"val_189"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    284 |  # node_Reshape_284
           %"val_190"<?,?> ⬅️ ::Reshape(%"add_613", %"val_189") {allowzero=0}
    285 |  # node_Concat_285
           %"val_191"<?,?> ⬅️ ::Concat(%"val_190", %"val_87", %"val_98", %"val_99") {axis=0}
    286 |  # node_Cast_286
           %"val_192"<?,?> ⬅️ ::Cast(%"val_191") {to=INT64}
    287 |  # node_Reshape_287
           %"view_27"<FLOAT,[s0 + 131,1,3,128]> ⬅️ ::Reshape(%"linear_8", %"val_192") {allowzero=True}
    288 |  # node_Unsqueeze_288
           %"unsqueeze_6"<FLOAT,[1,s0 + 131,1,3,128]> ⬅️ ::Unsqueeze(%"view_27", %"val_10")
    289 |  # node_Transpose_289
           %"transpose_13"<FLOAT,[3,s0 + 131,1,1,128]> ⬅️ ::Transpose(%"unsqueeze_6") {perm=[3, 1, 2, 0, 4]}
    290 |  # node_Squeeze_290
           %"squeeze_3"<FLOAT,[3,s0 + 131,1,128]> ⬅️ ::Squeeze(%"transpose_13", %"val_11")
    291 |  # node_Identity_291
           %"clone_9"<FLOAT,[3,s0 + 131,1,128]> ⬅️ ::Identity(%"squeeze_3")
    292 |  # node_Gather_292
           %"select_6"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_9", %"val_1") {axis=0}
    293 |  # node_Gather_293
           %"select_7"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_9", %"val_12") {axis=0}
    294 |  # node_Gather_294
           %"select_8"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_9", %"val_13") {axis=0}
    295 |  # node_Constant_295
           %"val_193"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    296 |  # node_Reshape_296
           %"val_194"<?,?> ⬅️ ::Reshape(%"add_613", %"val_193") {allowzero=0}
    297 |  # node_Concat_297
           %"val_195"<?,?> ⬅️ ::Concat(%"val_194", %"val_104", %"val_105") {axis=0}
    298 |  # node_Cast_298
           %"val_196"<?,?> ⬅️ ::Cast(%"val_195") {to=INT64}
    299 |  # node_Reshape_299
           %"view_28"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_6", %"val_196") {allowzero=True}
    300 |  # node_Transpose_300
           %"transpose_14"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_28") {perm=[1, 0, 2]}
    301 |  # node_Constant_301
           %"val_197"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    302 |  # node_Reshape_302
           %"val_198"<?,?> ⬅️ ::Reshape(%"add_613", %"val_197") {allowzero=0}
    303 |  # node_Concat_303
           %"val_199"<?,?> ⬅️ ::Concat(%"val_198", %"val_104", %"val_105") {axis=0}
    304 |  # node_Cast_304
           %"val_200"<?,?> ⬅️ ::Cast(%"val_199") {to=INT64}
    305 |  # node_Reshape_305
           %"view_29"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_7", %"val_200") {allowzero=True}
    306 |  # node_Transpose_306
           %"transpose_15"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_29") {perm=[1, 0, 2]}
    307 |  # node_Constant_307
           %"val_201"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    308 |  # node_Reshape_308
           %"val_202"<?,?> ⬅️ ::Reshape(%"add_613", %"val_201") {allowzero=0}
    309 |  # node_Concat_309
           %"val_203"<?,?> ⬅️ ::Concat(%"val_202", %"val_104", %"val_105") {axis=0}
    310 |  # node_Cast_310
           %"val_204"<?,?> ⬅️ ::Cast(%"val_203") {to=INT64}
    311 |  # node_Reshape_311
           %"view_30"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_8", %"val_204") {allowzero=True}
    312 |  # node_Transpose_312
           %"transpose_16"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_30") {perm=[1, 0, 2]}
    313 |  # node_Constant_313
           %"val_205"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    314 |  # node_Reshape_314
           %"val_206"<?,?> ⬅️ ::Reshape(%"add_613", %"val_205") {allowzero=0}
    315 |  # node_Concat_315
           %"val_207"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_206", %"val_105") {axis=0}
    316 |  # node_Cast_316
           %"val_208"<?,?> ⬅️ ::Cast(%"val_207") {to=INT64}
    317 |  # node_Reshape_317
           %"view_34"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_14", %"val_208") {allowzero=True}
    318 |  # node_Constant_318
           %"val_209"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    319 |  # node_Reshape_319
           %"val_210"<?,?> ⬅️ ::Reshape(%"add_613", %"val_209") {allowzero=0}
    320 |  # node_Concat_320
           %"val_211"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_210", %"val_105") {axis=0}
    321 |  # node_Cast_321
           %"val_212"<?,?> ⬅️ ::Cast(%"val_211") {to=INT64}
    322 |  # node_Reshape_322
           %"view_35"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_15", %"val_212") {allowzero=True}
    323 |  # node_Constant_323
           %"val_213"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    324 |  # node_Reshape_324
           %"val_214"<?,?> ⬅️ ::Reshape(%"add_613", %"val_213") {allowzero=0}
    325 |  # node_Concat_325
           %"val_215"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_214", %"val_105") {axis=0}
    326 |  # node_Cast_326
           %"val_216"<?,?> ⬅️ ::Cast(%"val_215") {to=INT64}
    327 |  # node_Reshape_327
           %"view_36"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_16", %"val_216") {allowzero=True}
    328 |  # node_Constant_328
           %"val_217"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    329 |  # node_Reshape_329
           %"val_218"<?,?> ⬅️ ::Reshape(%"add_613", %"val_217") {allowzero=0}
    330 |  # node_Concat_330
           %"val_219"<?,?> ⬅️ ::Concat(%"val_87", %"val_87", %"val_87", %"val_218") {axis=0}
    331 |  # node_Cast_331
           %"val_220"<?,?> ⬅️ ::Cast(%"val_219") {to=INT64}
    332 |  # node_Reshape_332
           %"view_37"<FLOAT,[1,1,1,s0 + 131]> ⬅️ ::Reshape(%"masked_fill_2", %"val_220") {allowzero=True}
    333 |  # node_Cast_333
           %"val_221"<?,?> ⬅️ ::Cast(%"val_132") {to=INT64}
    334 |  # node_Abs_334
           %"val_222"<?,?> ⬅️ ::Abs(%"val_221")
    335 |  # node_Expand_335
           %"expand_5"<FLOAT,[1,4,1,s0 + 131]> ⬅️ ::Expand(%"view_37", %"val_222")
    336 |  # node_Constant_336
           %"val_223"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    337 |  # node_Reshape_337
           %"val_224"<?,?> ⬅️ ::Reshape(%"add_613", %"val_223") {allowzero=0}
    338 |  # node_Concat_338
           %"val_225"<?,?> ⬅️ ::Concat(%"val_104", %"val_87", %"val_224") {axis=0}
    339 |  # node_Cast_339
           %"val_226"<?,?> ⬅️ ::Cast(%"val_225") {to=INT64}
    340 |  # node_Reshape_340
           %"view_38"<FLOAT,[4,1,s0 + 131]> ⬅️ ::Reshape(%"expand_5", %"val_226") {allowzero=True}
    341 |  # node_Constant_341
           %"val_227"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    342 |  # node_Reshape_342
           %"val_228"<?,?> ⬅️ ::Reshape(%"add_613", %"val_227") {allowzero=0}
    343 |  # node_Concat_343
           %"val_229"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_42", %"val_228") {axis=0}
    344 |  # node_Cast_344
           %"val_230"<?,?> ⬅️ ::Cast(%"val_229") {to=INT64}
    345 |  # node_Reshape_345
           %"view_39"<FLOAT,[1,4,1,s0 + 131]> ⬅️ ::Reshape(%"view_38", %"val_230") {allowzero=True}
    346 |  # node_Shape_346
           %"val_231"<?,?> ⬅️ ::Shape(%"view_34") {start=0}
    347 |  # node_Constant_347
           %"val_232"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    348 |  # node_Gather_348
           %"val_233"<?,?> ⬅️ ::Gather(%"val_231", %"val_232") {axis=0}
    349 |  # node_CastLike_349
           %"val_234"<?,?> ⬅️ ::CastLike(%"val_233", %"view_34")
    350 |  # node_Constant_350
           %"val_235"<?,?> ⬅️ ::Constant() {value_float=1.0}
    351 |  # node_CastLike_351
           %"val_236"<?,?> ⬅️ ::CastLike(%"val_235", %"view_34")
    352 |  # node_Sqrt_352
           %"val_237"<?,?> ⬅️ ::Sqrt(%"val_234")
    353 |  # node_Div_353
           %"val_238"<?,?> ⬅️ ::Div(%"val_236", %"val_237")
    354 |  # node_CastLike_354
           %"val_239"<?,?> ⬅️ ::CastLike(%"val_238", %"view_34")
    355 |  # node_Shape_355
           %"val_240"<?,?> ⬅️ ::Shape(%"view_35") {start=0}
    356 |  # node_Constant_356
           %"val_241"<?,?> ⬅️ ::Constant() {value_ints=[9223372036854775807]}
    357 |  # node_Slice_357
           %"val_242"<?,?> ⬅️ ::Slice(%"val_240", %"val_42", %"val_241")
    358 |  # node_Slice_358
           %"val_243"<?,?> ⬅️ ::Slice(%"val_240", %"val_11", %"val_42")
    359 |  # node_Constant_359
           %"val_244"<?,?> ⬅️ ::Constant() {value_ints=[-9223372036854775808]}
    360 |  # node_Slice_360
           %"val_245"<?,?> ⬅️ ::Slice(%"val_240", %"val_244", %"val_11")
    361 |  # node_Constant_361
           %"val_246"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    362 |  # node_Concat_362
           %"val_247"<?,?> ⬅️ ::Concat(%"val_246", %"val_243", %"val_242") {axis=0}
    363 |  # node_Reshape_363
           %"val_248"<?,?> ⬅️ ::Reshape(%"view_35", %"val_247") {allowzero=0}
    364 |  # node_Transpose_364
           %"val_249"<?,?> ⬅️ ::Transpose(%"val_248") {perm=[0, 2, 1]}
    365 |  # node_Concat_365
           %"val_250"<?,?> ⬅️ ::Concat(%"val_245", %"val_242", %"val_243") {axis=0}
    366 |  # node_Reshape_366
           %"val_251"<?,?> ⬅️ ::Reshape(%"val_249", %"val_250") {allowzero=0}
    367 |  # node_Sqrt_367
           %"val_252"<?,?> ⬅️ ::Sqrt(%"val_239")
    368 |  # node_Mul_368
           %"val_253"<?,?> ⬅️ ::Mul(%"view_34", %"val_252")
    369 |  # node_Sqrt_369
           %"val_254"<?,?> ⬅️ ::Sqrt(%"val_239")
    370 |  # node_Mul_370
           %"val_255"<?,?> ⬅️ ::Mul(%"val_251", %"val_254")
    371 |  # node_MatMul_371
           %"val_256"<?,?> ⬅️ ::MatMul(%"val_253", %"val_255")
    372 |  # node_Add_372
           %"val_257"<?,?> ⬅️ ::Add(%"val_256", %"view_39")
    373 |  # node_Softmax_373
           %"val_258"<?,?> ⬅️ ::Softmax(%"val_257") {axis=-1}
    374 |  # node_MatMul_374
           %"scaled_dot_product_attention_2"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::MatMul(%"val_258", %"view_36")
    375 |  # node_Transpose_375
           %"permute_2"<FLOAT,[s0 + 131,1,4,32]> ⬅️ ::Transpose(%"scaled_dot_product_attention_2") {perm=[2, 0, 1, 3]}
    376 |  # node_Constant_376
           %"val_259"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    377 |  # node_Reshape_377
           %"val_260"<?,?> ⬅️ ::Reshape(%"add_613", %"val_259") {allowzero=0}
    378 |  # node_Concat_378
           %"val_261"<?,?> ⬅️ ::Concat(%"val_260", %"val_99") {axis=0}
    379 |  # node_Cast_379
           %"val_262"<?,?> ⬅️ ::Cast(%"val_261") {to=INT64}
    380 |  # node_Reshape_380
           %"view_40"<FLOAT,[s0 + 131,128]> ⬅️ ::Reshape(%"permute_2", %"val_262") {allowzero=True}
    381 |  # node_Gemm_381
           %"linear_9"<FLOAT,[s0 + 131,128]> ⬅️ ::Gemm(%"view_40", %"core.mol_encoder.layers.1.self_attn.out_proj.weight"{...}, %"core.mol_encoder.layers.1.self_attn.out_proj.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    382 |  # node_Constant_382
           %"val_263"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    383 |  # node_Reshape_383
           %"val_264"<?,?> ⬅️ ::Reshape(%"add_613", %"val_263") {allowzero=0}
    384 |  # node_Concat_384
           %"val_265"<?,?> ⬅️ ::Concat(%"val_264", %"val_87", %"val_99") {axis=0}
    385 |  # node_Cast_385
           %"val_266"<?,?> ⬅️ ::Cast(%"val_265") {to=INT64}
    386 |  # node_Reshape_386
           %"view_41"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Reshape(%"linear_9", %"val_266") {allowzero=True}
    387 |  # node_Transpose_387
           %"transpose_17"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Transpose(%"view_41") {perm=[1, 0, 2]}
    388 |  # node_Identity_388
           %"clone_10"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Identity(%"transpose_17")
    389 |  # node_Add_389
           %"add_265"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"layer_norm_3", %"clone_10")
    390 |  # node_LayerNormalization_390
           %"layer_norm_4"<FLOAT,[1,s0 + 131,128]>, %"val_267"<?,?>, %"val_268"<?,?> ⬅️ ::LayerNormalization(%"add_265", %"core.mol_encoder.layers.1.norm1.weight"{...}, %"core.mol_encoder.layers.1.norm1.bias"{...}) {stash_type=1, epsilon=1e-05, axis=-1}
    391 |  # node_Transpose_391
           %"val_269"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.1.linear1.weight"{...}) {perm=[1, 0]}
    392 |  # node_MatMul_392
           %"val_270"<?,?> ⬅️ ::MatMul(%"layer_norm_4", %"val_269")
    393 |  # node_Add_393
           %"linear_10"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Add(%"val_270", %"core.mol_encoder.layers.1.linear1.bias"{...})
    394 |  # node_Relu_394
           %"relu_2"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Relu(%"linear_10")
    395 |  # node_Identity_395
           %"clone_11"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Identity(%"relu_2")
    396 |  # node_Transpose_396
           %"val_271"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.1.linear2.weight"{...}) {perm=[1, 0]}
    397 |  # node_MatMul_397
           %"val_272"<?,?> ⬅️ ::MatMul(%"clone_11", %"val_271")
    398 |  # node_Add_398
           %"linear_11"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"val_272", %"core.mol_encoder.layers.1.linear2.bias"{...})
    399 |  # node_Identity_399
           %"clone_12"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Identity(%"linear_11")
    400 |  # node_Add_400
           %"add_287"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"layer_norm_4", %"clone_12")
    401 |  # node_LayerNormalization_401
           %"layer_norm_5"<FLOAT,[1,s0 + 131,128]>, %"val_273"<?,?>, %"val_274"<?,?> ⬅️ ::LayerNormalization(%"add_287", %"core.mol_encoder.layers.1.norm2.weight"{...}, %"core.mol_encoder.layers.1.norm2.bias"{...}) {stash_type=1, epsilon=1e-05, axis=-1}
    402 |  # node_Transpose_402
           %"transpose_18"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Transpose(%"layer_norm_5") {perm=[1, 0, 2]}
    403 |  # node_Transpose_403
           %"val_275"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.2.self_attn.in_proj_weight"{...}) {perm=[1, 0]}
    404 |  # node_MatMul_404
           %"val_276"<?,?> ⬅️ ::MatMul(%"transpose_18", %"val_275")
    405 |  # node_Add_405
           %"linear_12"<FLOAT,[s0 + 131,1,384]> ⬅️ ::Add(%"val_276", %"core.mol_encoder.layers.2.self_attn.in_proj_bias"{...})
    406 |  # node_Constant_406
           %"val_277"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    407 |  # node_Reshape_407
           %"val_278"<?,?> ⬅️ ::Reshape(%"add_613", %"val_277") {allowzero=0}
    408 |  # node_Concat_408
           %"val_279"<?,?> ⬅️ ::Concat(%"val_278", %"val_87", %"val_98", %"val_99") {axis=0}
    409 |  # node_Cast_409
           %"val_280"<?,?> ⬅️ ::Cast(%"val_279") {to=INT64}
    410 |  # node_Reshape_410
           %"view_42"<FLOAT,[s0 + 131,1,3,128]> ⬅️ ::Reshape(%"linear_12", %"val_280") {allowzero=True}
    411 |  # node_Unsqueeze_411
           %"unsqueeze_7"<FLOAT,[1,s0 + 131,1,3,128]> ⬅️ ::Unsqueeze(%"view_42", %"val_10")
    412 |  # node_Transpose_412
           %"transpose_19"<FLOAT,[3,s0 + 131,1,1,128]> ⬅️ ::Transpose(%"unsqueeze_7") {perm=[3, 1, 2, 0, 4]}
    413 |  # node_Squeeze_413
           %"squeeze_4"<FLOAT,[3,s0 + 131,1,128]> ⬅️ ::Squeeze(%"transpose_19", %"val_11")
    414 |  # node_Identity_414
           %"clone_13"<FLOAT,[3,s0 + 131,1,128]> ⬅️ ::Identity(%"squeeze_4")
    415 |  # node_Gather_415
           %"select_9"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_13", %"val_1") {axis=0}
    416 |  # node_Gather_416
           %"select_10"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_13", %"val_12") {axis=0}
    417 |  # node_Gather_417
           %"select_11"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_13", %"val_13") {axis=0}
    418 |  # node_Constant_418
           %"val_281"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    419 |  # node_Reshape_419
           %"val_282"<?,?> ⬅️ ::Reshape(%"add_613", %"val_281") {allowzero=0}
    420 |  # node_Concat_420
           %"val_283"<?,?> ⬅️ ::Concat(%"val_282", %"val_104", %"val_105") {axis=0}
    421 |  # node_Cast_421
           %"val_284"<?,?> ⬅️ ::Cast(%"val_283") {to=INT64}
    422 |  # node_Reshape_422
           %"view_43"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_9", %"val_284") {allowzero=True}
    423 |  # node_Transpose_423
           %"transpose_20"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_43") {perm=[1, 0, 2]}
    424 |  # node_Constant_424
           %"val_285"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    425 |  # node_Reshape_425
           %"val_286"<?,?> ⬅️ ::Reshape(%"add_613", %"val_285") {allowzero=0}
    426 |  # node_Concat_426
           %"val_287"<?,?> ⬅️ ::Concat(%"val_286", %"val_104", %"val_105") {axis=0}
    427 |  # node_Cast_427
           %"val_288"<?,?> ⬅️ ::Cast(%"val_287") {to=INT64}
    428 |  # node_Reshape_428
           %"view_44"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_10", %"val_288") {allowzero=True}
    429 |  # node_Transpose_429
           %"transpose_21"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_44") {perm=[1, 0, 2]}
    430 |  # node_Constant_430
           %"val_289"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    431 |  # node_Reshape_431
           %"val_290"<?,?> ⬅️ ::Reshape(%"add_613", %"val_289") {allowzero=0}
    432 |  # node_Concat_432
           %"val_291"<?,?> ⬅️ ::Concat(%"val_290", %"val_104", %"val_105") {axis=0}
    433 |  # node_Cast_433
           %"val_292"<?,?> ⬅️ ::Cast(%"val_291") {to=INT64}
    434 |  # node_Reshape_434
           %"view_45"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_11", %"val_292") {allowzero=True}
    435 |  # node_Transpose_435
           %"transpose_22"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_45") {perm=[1, 0, 2]}
    436 |  # node_Constant_436
           %"val_293"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    437 |  # node_Reshape_437
           %"val_294"<?,?> ⬅️ ::Reshape(%"add_613", %"val_293") {allowzero=0}
    438 |  # node_Concat_438
           %"val_295"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_294", %"val_105") {axis=0}
    439 |  # node_Cast_439
           %"val_296"<?,?> ⬅️ ::Cast(%"val_295") {to=INT64}
    440 |  # node_Reshape_440
           %"view_49"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_20", %"val_296") {allowzero=True}
    441 |  # node_Constant_441
           %"val_297"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    442 |  # node_Reshape_442
           %"val_298"<?,?> ⬅️ ::Reshape(%"add_613", %"val_297") {allowzero=0}
    443 |  # node_Concat_443
           %"val_299"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_298", %"val_105") {axis=0}
    444 |  # node_Cast_444
           %"val_300"<?,?> ⬅️ ::Cast(%"val_299") {to=INT64}
    445 |  # node_Reshape_445
           %"view_50"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_21", %"val_300") {allowzero=True}
    446 |  # node_Constant_446
           %"val_301"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    447 |  # node_Reshape_447
           %"val_302"<?,?> ⬅️ ::Reshape(%"add_613", %"val_301") {allowzero=0}
    448 |  # node_Concat_448
           %"val_303"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_302", %"val_105") {axis=0}
    449 |  # node_Cast_449
           %"val_304"<?,?> ⬅️ ::Cast(%"val_303") {to=INT64}
    450 |  # node_Reshape_450
           %"view_51"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_22", %"val_304") {allowzero=True}
    451 |  # node_Constant_451
           %"val_305"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    452 |  # node_Reshape_452
           %"val_306"<?,?> ⬅️ ::Reshape(%"add_613", %"val_305") {allowzero=0}
    453 |  # node_Concat_453
           %"val_307"<?,?> ⬅️ ::Concat(%"val_87", %"val_87", %"val_87", %"val_306") {axis=0}
    454 |  # node_Cast_454
           %"val_308"<?,?> ⬅️ ::Cast(%"val_307") {to=INT64}
    455 |  # node_Reshape_455
           %"view_52"<FLOAT,[1,1,1,s0 + 131]> ⬅️ ::Reshape(%"masked_fill_2", %"val_308") {allowzero=True}
    456 |  # node_Cast_456
           %"val_309"<?,?> ⬅️ ::Cast(%"val_132") {to=INT64}
    457 |  # node_Abs_457
           %"val_310"<?,?> ⬅️ ::Abs(%"val_309")
    458 |  # node_Expand_458
           %"expand_7"<FLOAT,[1,4,1,s0 + 131]> ⬅️ ::Expand(%"view_52", %"val_310")
    459 |  # node_Constant_459
           %"val_311"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    460 |  # node_Reshape_460
           %"val_312"<?,?> ⬅️ ::Reshape(%"add_613", %"val_311") {allowzero=0}
    461 |  # node_Concat_461
           %"val_313"<?,?> ⬅️ ::Concat(%"val_104", %"val_87", %"val_312") {axis=0}
    462 |  # node_Cast_462
           %"val_314"<?,?> ⬅️ ::Cast(%"val_313") {to=INT64}
    463 |  # node_Reshape_463
           %"view_53"<FLOAT,[4,1,s0 + 131]> ⬅️ ::Reshape(%"expand_7", %"val_314") {allowzero=True}
    464 |  # node_Constant_464
           %"val_315"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    465 |  # node_Reshape_465
           %"val_316"<?,?> ⬅️ ::Reshape(%"add_613", %"val_315") {allowzero=0}
    466 |  # node_Concat_466
           %"val_317"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_42", %"val_316") {axis=0}
    467 |  # node_Cast_467
           %"val_318"<?,?> ⬅️ ::Cast(%"val_317") {to=INT64}
    468 |  # node_Reshape_468
           %"view_54"<FLOAT,[1,4,1,s0 + 131]> ⬅️ ::Reshape(%"view_53", %"val_318") {allowzero=True}
    469 |  # node_Shape_469
           %"val_319"<?,?> ⬅️ ::Shape(%"view_49") {start=0}
    470 |  # node_Constant_470
           %"val_320"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    471 |  # node_Gather_471
           %"val_321"<?,?> ⬅️ ::Gather(%"val_319", %"val_320") {axis=0}
    472 |  # node_CastLike_472
           %"val_322"<?,?> ⬅️ ::CastLike(%"val_321", %"view_49")
    473 |  # node_Constant_473
           %"val_323"<?,?> ⬅️ ::Constant() {value_float=1.0}
    474 |  # node_CastLike_474
           %"val_324"<?,?> ⬅️ ::CastLike(%"val_323", %"view_49")
    475 |  # node_Sqrt_475
           %"val_325"<?,?> ⬅️ ::Sqrt(%"val_322")
    476 |  # node_Div_476
           %"val_326"<?,?> ⬅️ ::Div(%"val_324", %"val_325")
    477 |  # node_CastLike_477
           %"val_327"<?,?> ⬅️ ::CastLike(%"val_326", %"view_49")
    478 |  # node_Shape_478
           %"val_328"<?,?> ⬅️ ::Shape(%"view_50") {start=0}
    479 |  # node_Constant_479
           %"val_329"<?,?> ⬅️ ::Constant() {value_ints=[9223372036854775807]}
    480 |  # node_Slice_480
           %"val_330"<?,?> ⬅️ ::Slice(%"val_328", %"val_42", %"val_329")
    481 |  # node_Slice_481
           %"val_331"<?,?> ⬅️ ::Slice(%"val_328", %"val_11", %"val_42")
    482 |  # node_Constant_482
           %"val_332"<?,?> ⬅️ ::Constant() {value_ints=[-9223372036854775808]}
    483 |  # node_Slice_483
           %"val_333"<?,?> ⬅️ ::Slice(%"val_328", %"val_332", %"val_11")
    484 |  # node_Constant_484
           %"val_334"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    485 |  # node_Concat_485
           %"val_335"<?,?> ⬅️ ::Concat(%"val_334", %"val_331", %"val_330") {axis=0}
    486 |  # node_Reshape_486
           %"val_336"<?,?> ⬅️ ::Reshape(%"view_50", %"val_335") {allowzero=0}
    487 |  # node_Transpose_487
           %"val_337"<?,?> ⬅️ ::Transpose(%"val_336") {perm=[0, 2, 1]}
    488 |  # node_Concat_488
           %"val_338"<?,?> ⬅️ ::Concat(%"val_333", %"val_330", %"val_331") {axis=0}
    489 |  # node_Reshape_489
           %"val_339"<?,?> ⬅️ ::Reshape(%"val_337", %"val_338") {allowzero=0}
    490 |  # node_Sqrt_490
           %"val_340"<?,?> ⬅️ ::Sqrt(%"val_327")
    491 |  # node_Mul_491
           %"val_341"<?,?> ⬅️ ::Mul(%"view_49", %"val_340")
    492 |  # node_Sqrt_492
           %"val_342"<?,?> ⬅️ ::Sqrt(%"val_327")
    493 |  # node_Mul_493
           %"val_343"<?,?> ⬅️ ::Mul(%"val_339", %"val_342")
    494 |  # node_MatMul_494
           %"val_344"<?,?> ⬅️ ::MatMul(%"val_341", %"val_343")
    495 |  # node_Add_495
           %"val_345"<?,?> ⬅️ ::Add(%"val_344", %"view_54")
    496 |  # node_Softmax_496
           %"val_346"<?,?> ⬅️ ::Softmax(%"val_345") {axis=-1}
    497 |  # node_MatMul_497
           %"scaled_dot_product_attention_3"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::MatMul(%"val_346", %"view_51")
    498 |  # node_Transpose_498
           %"permute_3"<FLOAT,[s0 + 131,1,4,32]> ⬅️ ::Transpose(%"scaled_dot_product_attention_3") {perm=[2, 0, 1, 3]}
    499 |  # node_Constant_499
           %"val_347"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    500 |  # node_Reshape_500
           %"val_348"<?,?> ⬅️ ::Reshape(%"add_613", %"val_347") {allowzero=0}
    501 |  # node_Concat_501
           %"val_349"<?,?> ⬅️ ::Concat(%"val_348", %"val_99") {axis=0}
    502 |  # node_Cast_502
           %"val_350"<?,?> ⬅️ ::Cast(%"val_349") {to=INT64}
    503 |  # node_Reshape_503
           %"view_55"<FLOAT,[s0 + 131,128]> ⬅️ ::Reshape(%"permute_3", %"val_350") {allowzero=True}
    504 |  # node_Gemm_504
           %"linear_13"<FLOAT,[s0 + 131,128]> ⬅️ ::Gemm(%"view_55", %"core.mol_encoder.layers.2.self_attn.out_proj.weight"{...}, %"core.mol_encoder.layers.2.self_attn.out_proj.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    505 |  # node_Constant_505
           %"val_351"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    506 |  # node_Reshape_506
           %"val_352"<?,?> ⬅️ ::Reshape(%"add_613", %"val_351") {allowzero=0}
    507 |  # node_Concat_507
           %"val_353"<?,?> ⬅️ ::Concat(%"val_352", %"val_87", %"val_99") {axis=0}
    508 |  # node_Cast_508
           %"val_354"<?,?> ⬅️ ::Cast(%"val_353") {to=INT64}
    509 |  # node_Reshape_509
           %"view_56"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Reshape(%"linear_13", %"val_354") {allowzero=True}
    510 |  # node_Transpose_510
           %"transpose_23"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Transpose(%"view_56") {perm=[1, 0, 2]}
    511 |  # node_Identity_511
           %"clone_14"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Identity(%"transpose_23")
    512 |  # node_Add_512
           %"add_400"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"layer_norm_5", %"clone_14")
    513 |  # node_LayerNormalization_513
           %"layer_norm_6"<FLOAT,[1,s0 + 131,128]>, %"val_355"<?,?>, %"val_356"<?,?> ⬅️ ::LayerNormalization(%"add_400", %"core.mol_encoder.layers.2.norm1.weight"{...}, %"core.mol_encoder.layers.2.norm1.bias"{...}) {stash_type=1, epsilon=1e-05, axis=-1}
    514 |  # node_Transpose_514
           %"val_357"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.2.linear1.weight"{...}) {perm=[1, 0]}
    515 |  # node_MatMul_515
           %"val_358"<?,?> ⬅️ ::MatMul(%"layer_norm_6", %"val_357")
    516 |  # node_Add_516
           %"linear_14"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Add(%"val_358", %"core.mol_encoder.layers.2.linear1.bias"{...})
    517 |  # node_Relu_517
           %"relu_3"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Relu(%"linear_14")
    518 |  # node_Identity_518
           %"clone_15"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Identity(%"relu_3")
    519 |  # node_Transpose_519
           %"val_359"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.2.linear2.weight"{...}) {perm=[1, 0]}
    520 |  # node_MatMul_520
           %"val_360"<?,?> ⬅️ ::MatMul(%"clone_15", %"val_359")
    521 |  # node_Add_521
           %"linear_15"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"val_360", %"core.mol_encoder.layers.2.linear2.bias"{...})
    522 |  # node_Identity_522
           %"clone_16"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Identity(%"linear_15")
    523 |  # node_Add_523
           %"add_422"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"layer_norm_6", %"clone_16")
    524 |  # node_LayerNormalization_524
           %"layer_norm_7"<FLOAT,[1,s0 + 131,128]>, %"val_361"<?,?>, %"val_362"<?,?> ⬅️ ::LayerNormalization(%"add_422", %"core.mol_encoder.layers.2.norm2.weight"{...}, %"core.mol_encoder.layers.2.norm2.bias"{...}) {stash_type=1, epsilon=1e-05, axis=-1}
    525 |  # node_Transpose_525
           %"transpose_24"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Transpose(%"layer_norm_7") {perm=[1, 0, 2]}
    526 |  # node_Transpose_526
           %"val_363"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.3.self_attn.in_proj_weight"{...}) {perm=[1, 0]}
    527 |  # node_MatMul_527
           %"val_364"<?,?> ⬅️ ::MatMul(%"transpose_24", %"val_363")
    528 |  # node_Add_528
           %"linear_16"<FLOAT,[s0 + 131,1,384]> ⬅️ ::Add(%"val_364", %"core.mol_encoder.layers.3.self_attn.in_proj_bias"{...})
    529 |  # node_Constant_529
           %"val_365"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    530 |  # node_Reshape_530
           %"val_366"<?,?> ⬅️ ::Reshape(%"add_613", %"val_365") {allowzero=0}
    531 |  # node_Concat_531
           %"val_367"<?,?> ⬅️ ::Concat(%"val_366", %"val_87", %"val_98", %"val_99") {axis=0}
    532 |  # node_Cast_532
           %"val_368"<?,?> ⬅️ ::Cast(%"val_367") {to=INT64}
    533 |  # node_Reshape_533
           %"view_57"<FLOAT,[s0 + 131,1,3,128]> ⬅️ ::Reshape(%"linear_16", %"val_368") {allowzero=True}
    534 |  # node_Unsqueeze_534
           %"unsqueeze_8"<FLOAT,[1,s0 + 131,1,3,128]> ⬅️ ::Unsqueeze(%"view_57", %"val_10")
    535 |  # node_Transpose_535
           %"transpose_25"<FLOAT,[3,s0 + 131,1,1,128]> ⬅️ ::Transpose(%"unsqueeze_8") {perm=[3, 1, 2, 0, 4]}
    536 |  # node_Squeeze_536
           %"squeeze_5"<FLOAT,[3,s0 + 131,1,128]> ⬅️ ::Squeeze(%"transpose_25", %"val_11")
    537 |  # node_Identity_537
           %"clone_17"<FLOAT,[3,s0 + 131,1,128]> ⬅️ ::Identity(%"squeeze_5")
    538 |  # node_Gather_538
           %"select_12"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_17", %"val_1") {axis=0}
    539 |  # node_Gather_539
           %"select_13"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_17", %"val_12") {axis=0}
    540 |  # node_Gather_540
           %"select_14"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Gather(%"clone_17", %"val_13") {axis=0}
    541 |  # node_Constant_541
           %"val_369"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    542 |  # node_Reshape_542
           %"val_370"<?,?> ⬅️ ::Reshape(%"add_613", %"val_369") {allowzero=0}
    543 |  # node_Concat_543
           %"val_371"<?,?> ⬅️ ::Concat(%"val_370", %"val_104", %"val_105") {axis=0}
    544 |  # node_Cast_544
           %"val_372"<?,?> ⬅️ ::Cast(%"val_371") {to=INT64}
    545 |  # node_Reshape_545
           %"view_58"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_12", %"val_372") {allowzero=True}
    546 |  # node_Transpose_546
           %"transpose_26"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_58") {perm=[1, 0, 2]}
    547 |  # node_Constant_547
           %"val_373"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    548 |  # node_Reshape_548
           %"val_374"<?,?> ⬅️ ::Reshape(%"add_613", %"val_373") {allowzero=0}
    549 |  # node_Concat_549
           %"val_375"<?,?> ⬅️ ::Concat(%"val_374", %"val_104", %"val_105") {axis=0}
    550 |  # node_Cast_550
           %"val_376"<?,?> ⬅️ ::Cast(%"val_375") {to=INT64}
    551 |  # node_Reshape_551
           %"view_59"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_13", %"val_376") {allowzero=True}
    552 |  # node_Transpose_552
           %"transpose_27"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_59") {perm=[1, 0, 2]}
    553 |  # node_Constant_553
           %"val_377"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    554 |  # node_Reshape_554
           %"val_378"<?,?> ⬅️ ::Reshape(%"add_613", %"val_377") {allowzero=0}
    555 |  # node_Concat_555
           %"val_379"<?,?> ⬅️ ::Concat(%"val_378", %"val_104", %"val_105") {axis=0}
    556 |  # node_Cast_556
           %"val_380"<?,?> ⬅️ ::Cast(%"val_379") {to=INT64}
    557 |  # node_Reshape_557
           %"view_60"<FLOAT,[s0 + 131,4,32]> ⬅️ ::Reshape(%"select_14", %"val_380") {allowzero=True}
    558 |  # node_Transpose_558
           %"transpose_28"<FLOAT,[4,s0 + 131,32]> ⬅️ ::Transpose(%"view_60") {perm=[1, 0, 2]}
    559 |  # node_Constant_559
           %"val_381"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    560 |  # node_Reshape_560
           %"val_382"<?,?> ⬅️ ::Reshape(%"add_613", %"val_381") {allowzero=0}
    561 |  # node_Concat_561
           %"val_383"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_382", %"val_105") {axis=0}
    562 |  # node_Cast_562
           %"val_384"<?,?> ⬅️ ::Cast(%"val_383") {to=INT64}
    563 |  # node_Reshape_563
           %"view_64"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_26", %"val_384") {allowzero=True}
    564 |  # node_Constant_564
           %"val_385"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    565 |  # node_Reshape_565
           %"val_386"<?,?> ⬅️ ::Reshape(%"add_613", %"val_385") {allowzero=0}
    566 |  # node_Concat_566
           %"val_387"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_386", %"val_105") {axis=0}
    567 |  # node_Cast_567
           %"val_388"<?,?> ⬅️ ::Cast(%"val_387") {to=INT64}
    568 |  # node_Reshape_568
           %"view_65"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_27", %"val_388") {allowzero=True}
    569 |  # node_Constant_569
           %"val_389"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    570 |  # node_Reshape_570
           %"val_390"<?,?> ⬅️ ::Reshape(%"add_613", %"val_389") {allowzero=0}
    571 |  # node_Concat_571
           %"val_391"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_390", %"val_105") {axis=0}
    572 |  # node_Cast_572
           %"val_392"<?,?> ⬅️ ::Cast(%"val_391") {to=INT64}
    573 |  # node_Reshape_573
           %"view_66"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::Reshape(%"transpose_28", %"val_392") {allowzero=True}
    574 |  # node_Constant_574
           %"val_393"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    575 |  # node_Reshape_575
           %"val_394"<?,?> ⬅️ ::Reshape(%"add_613", %"val_393") {allowzero=0}
    576 |  # node_Concat_576
           %"val_395"<?,?> ⬅️ ::Concat(%"val_87", %"val_87", %"val_87", %"val_394") {axis=0}
    577 |  # node_Cast_577
           %"val_396"<?,?> ⬅️ ::Cast(%"val_395") {to=INT64}
    578 |  # node_Reshape_578
           %"view_67"<FLOAT,[1,1,1,s0 + 131]> ⬅️ ::Reshape(%"masked_fill_2", %"val_396") {allowzero=True}
    579 |  # node_Cast_579
           %"val_397"<?,?> ⬅️ ::Cast(%"val_132") {to=INT64}
    580 |  # node_Abs_580
           %"val_398"<?,?> ⬅️ ::Abs(%"val_397")
    581 |  # node_Expand_581
           %"expand_9"<FLOAT,[1,4,1,s0 + 131]> ⬅️ ::Expand(%"view_67", %"val_398")
    582 |  # node_Constant_582
           %"val_399"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    583 |  # node_Reshape_583
           %"val_400"<?,?> ⬅️ ::Reshape(%"add_613", %"val_399") {allowzero=0}
    584 |  # node_Concat_584
           %"val_401"<?,?> ⬅️ ::Concat(%"val_104", %"val_87", %"val_400") {axis=0}
    585 |  # node_Cast_585
           %"val_402"<?,?> ⬅️ ::Cast(%"val_401") {to=INT64}
    586 |  # node_Reshape_586
           %"view_68"<FLOAT,[4,1,s0 + 131]> ⬅️ ::Reshape(%"expand_9", %"val_402") {allowzero=True}
    587 |  # node_Constant_587
           %"val_403"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    588 |  # node_Reshape_588
           %"val_404"<?,?> ⬅️ ::Reshape(%"add_613", %"val_403") {allowzero=0}
    589 |  # node_Concat_589
           %"val_405"<?,?> ⬅️ ::Concat(%"val_87", %"val_104", %"val_42", %"val_404") {axis=0}
    590 |  # node_Cast_590
           %"val_406"<?,?> ⬅️ ::Cast(%"val_405") {to=INT64}
    591 |  # node_Reshape_591
           %"view_69"<FLOAT,[1,4,1,s0 + 131]> ⬅️ ::Reshape(%"view_68", %"val_406") {allowzero=True}
    592 |  # node_Shape_592
           %"val_407"<?,?> ⬅️ ::Shape(%"view_64") {start=0}
    593 |  # node_Constant_593
           %"val_408"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    594 |  # node_Gather_594
           %"val_409"<?,?> ⬅️ ::Gather(%"val_407", %"val_408") {axis=0}
    595 |  # node_CastLike_595
           %"val_410"<?,?> ⬅️ ::CastLike(%"val_409", %"view_64")
    596 |  # node_Constant_596
           %"val_411"<?,?> ⬅️ ::Constant() {value_float=1.0}
    597 |  # node_CastLike_597
           %"val_412"<?,?> ⬅️ ::CastLike(%"val_411", %"view_64")
    598 |  # node_Sqrt_598
           %"val_413"<?,?> ⬅️ ::Sqrt(%"val_410")
    599 |  # node_Div_599
           %"val_414"<?,?> ⬅️ ::Div(%"val_412", %"val_413")
    600 |  # node_CastLike_600
           %"val_415"<?,?> ⬅️ ::CastLike(%"val_414", %"view_64")
    601 |  # node_Shape_601
           %"val_416"<?,?> ⬅️ ::Shape(%"view_65") {start=0}
    602 |  # node_Constant_602
           %"val_417"<?,?> ⬅️ ::Constant() {value_ints=[9223372036854775807]}
    603 |  # node_Slice_603
           %"val_418"<?,?> ⬅️ ::Slice(%"val_416", %"val_42", %"val_417")
    604 |  # node_Slice_604
           %"val_419"<?,?> ⬅️ ::Slice(%"val_416", %"val_11", %"val_42")
    605 |  # node_Constant_605
           %"val_420"<?,?> ⬅️ ::Constant() {value_ints=[-9223372036854775808]}
    606 |  # node_Slice_606
           %"val_421"<?,?> ⬅️ ::Slice(%"val_416", %"val_420", %"val_11")
    607 |  # node_Constant_607
           %"val_422"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    608 |  # node_Concat_608
           %"val_423"<?,?> ⬅️ ::Concat(%"val_422", %"val_419", %"val_418") {axis=0}
    609 |  # node_Reshape_609
           %"val_424"<?,?> ⬅️ ::Reshape(%"view_65", %"val_423") {allowzero=0}
    610 |  # node_Transpose_610
           %"val_425"<?,?> ⬅️ ::Transpose(%"val_424") {perm=[0, 2, 1]}
    611 |  # node_Concat_611
           %"val_426"<?,?> ⬅️ ::Concat(%"val_421", %"val_418", %"val_419") {axis=0}
    612 |  # node_Reshape_612
           %"val_427"<?,?> ⬅️ ::Reshape(%"val_425", %"val_426") {allowzero=0}
    613 |  # node_Sqrt_613
           %"val_428"<?,?> ⬅️ ::Sqrt(%"val_415")
    614 |  # node_Mul_614
           %"val_429"<?,?> ⬅️ ::Mul(%"view_64", %"val_428")
    615 |  # node_Sqrt_615
           %"val_430"<?,?> ⬅️ ::Sqrt(%"val_415")
    616 |  # node_Mul_616
           %"val_431"<?,?> ⬅️ ::Mul(%"val_427", %"val_430")
    617 |  # node_MatMul_617
           %"val_432"<?,?> ⬅️ ::MatMul(%"val_429", %"val_431")
    618 |  # node_Add_618
           %"val_433"<?,?> ⬅️ ::Add(%"val_432", %"view_69")
    619 |  # node_Softmax_619
           %"val_434"<?,?> ⬅️ ::Softmax(%"val_433") {axis=-1}
    620 |  # node_MatMul_620
           %"scaled_dot_product_attention_4"<FLOAT,[1,4,s0 + 131,32]> ⬅️ ::MatMul(%"val_434", %"view_66")
    621 |  # node_Transpose_621
           %"permute_4"<FLOAT,[s0 + 131,1,4,32]> ⬅️ ::Transpose(%"scaled_dot_product_attention_4") {perm=[2, 0, 1, 3]}
    622 |  # node_Constant_622
           %"val_435"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    623 |  # node_Reshape_623
           %"val_436"<?,?> ⬅️ ::Reshape(%"add_613", %"val_435") {allowzero=0}
    624 |  # node_Concat_624
           %"val_437"<?,?> ⬅️ ::Concat(%"val_436", %"val_99") {axis=0}
    625 |  # node_Cast_625
           %"val_438"<?,?> ⬅️ ::Cast(%"val_437") {to=INT64}
    626 |  # node_Reshape_626
           %"view_70"<FLOAT,[s0 + 131,128]> ⬅️ ::Reshape(%"permute_4", %"val_438") {allowzero=True}
    627 |  # node_Gemm_627
           %"linear_17"<FLOAT,[s0 + 131,128]> ⬅️ ::Gemm(%"view_70", %"core.mol_encoder.layers.3.self_attn.out_proj.weight"{...}, %"core.mol_encoder.layers.3.self_attn.out_proj.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    628 |  # node_Constant_628
           %"val_439"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    629 |  # node_Reshape_629
           %"val_440"<?,?> ⬅️ ::Reshape(%"add_613", %"val_439") {allowzero=0}
    630 |  # node_Concat_630
           %"val_441"<?,?> ⬅️ ::Concat(%"val_440", %"val_87", %"val_99") {axis=0}
    631 |  # node_Cast_631
           %"val_442"<?,?> ⬅️ ::Cast(%"val_441") {to=INT64}
    632 |  # node_Reshape_632
           %"view_71"<FLOAT,[s0 + 131,1,128]> ⬅️ ::Reshape(%"linear_17", %"val_442") {allowzero=True}
    633 |  # node_Transpose_633
           %"transpose_29"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Transpose(%"view_71") {perm=[1, 0, 2]}
    634 |  # node_Identity_634
           %"clone_18"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Identity(%"transpose_29")
    635 |  # node_Add_635
           %"add_535"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"layer_norm_7", %"clone_18")
    636 |  # node_LayerNormalization_636
           %"layer_norm_8"<FLOAT,[1,s0 + 131,128]>, %"val_443"<?,?>, %"val_444"<?,?> ⬅️ ::LayerNormalization(%"add_535", %"core.mol_encoder.layers.3.norm1.weight"{...}, %"core.mol_encoder.layers.3.norm1.bias"{...}) {stash_type=1, epsilon=1e-05, axis=-1}
    637 |  # node_Transpose_637
           %"val_445"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.3.linear1.weight"{...}) {perm=[1, 0]}
    638 |  # node_MatMul_638
           %"val_446"<?,?> ⬅️ ::MatMul(%"layer_norm_8", %"val_445")
    639 |  # node_Add_639
           %"linear_18"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Add(%"val_446", %"core.mol_encoder.layers.3.linear1.bias"{...})
    640 |  # node_Relu_640
           %"relu_4"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Relu(%"linear_18")
    641 |  # node_Identity_641
           %"clone_19"<FLOAT,[1,s0 + 131,1024]> ⬅️ ::Identity(%"relu_4")
    642 |  # node_Transpose_642
           %"val_447"<?,?> ⬅️ ::Transpose(%"core.mol_encoder.layers.3.linear2.weight"{...}) {perm=[1, 0]}
    643 |  # node_MatMul_643
           %"val_448"<?,?> ⬅️ ::MatMul(%"clone_19", %"val_447")
    644 |  # node_Add_644
           %"linear_19"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"val_448", %"core.mol_encoder.layers.3.linear2.bias"{...})
    645 |  # node_Identity_645
           %"clone_20"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Identity(%"linear_19")
    646 |  # node_Add_646
           %"add_557"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Add(%"layer_norm_8", %"clone_20")
    647 |  # node_LayerNormalization_647
           %"layer_norm_9"<FLOAT,[1,s0 + 131,128]>, %"val_449"<?,?>, %"val_450"<?,?> ⬅️ ::LayerNormalization(%"add_557", %"core.mol_encoder.layers.3.norm2.weight"{...}, %"core.mol_encoder.layers.3.norm2.bias"{...}) {stash_type=1, epsilon=1e-05, axis=-1}
    648 |  # node_Unsqueeze_648
           %"unsqueeze_9"<BOOL,[1,s0 + 131,1]> ⬅️ ::Unsqueeze(%"unsqueeze_4", %"val_42")
    649 |  # node_CastLike_649
           %"val_451"<?,?> ⬅️ ::CastLike(%"val_72", %"layer_norm_9")
    650 |  # node_Where_650
           %"masked_fill_3"<FLOAT,[1,s0 + 131,128]> ⬅️ ::Where(%"unsqueeze_9", %"val_451", %"layer_norm_9")
    651 |  # node_Add_651
           %"add_576"<INT64,[]> ⬅️ ::Add(%"sym_size_int_17", %"val_12")
    652 |  # node_aten_unbind_652
           %"unbind"<?,?> ⬅️ pkg.onnxscript.torch_lib::aten_unbind(%"cbond_index") {dim=0}
    653 |  # node_aten_getitem_653
           %"getitem"<INT64,[s1]> ⬅️ pkg.onnxscript.torch_lib::aten_getitem(%"unbind", %"val_1")
    654 |  # node_aten_getitem_654
           %"getitem_1"<INT64,[s1]> ⬅️ pkg.onnxscript.torch_lib::aten_getitem(%"unbind", %"val_12")
    655 |  # node_Gather_655
           %"select_16"<FLOAT,[s0 + 131,128]> ⬅️ ::Gather(%"masked_fill_3", %"val_1") {axis=0}
    656 |  # node_Cast_656
           %"val_452"<?,?> ⬅️ ::Cast(%"val_12") {to=INT64}
    657 |  # node_Constant_657
           %"val_453"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    658 |  # node_Reshape_658
           %"val_454"<?,?> ⬅️ ::Reshape(%"val_452", %"val_453") {allowzero=0}
    659 |  # node_Cast_659
           %"val_455"<?,?> ⬅️ ::Cast(%"add_576") {to=INT64}
    660 |  # node_Constant_660
           %"val_456"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    661 |  # node_Reshape_661
           %"val_457"<?,?> ⬅️ ::Reshape(%"val_455", %"val_456") {allowzero=0}
    662 |  # node_Cast_662
           %"val_458"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
    663 |  # node_Constant_663
           %"val_459"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    664 |  # node_Reshape_664
           %"val_460"<?,?> ⬅️ ::Reshape(%"val_458", %"val_459") {allowzero=0}
    665 |  # node_Constant_665
           %"val_461"<?,?> ⬅️ ::Constant() {value_ints=[1]}
    666 |  # node_Slice_666
           %"slice_2"<FLOAT,[s0,128]> ⬅️ ::Slice(%"select_16", %"val_454", %"val_457", %"val_460", %"val_461")
    667 |  # node_Transpose_667
           %"val_462"<?,?> ⬅️ ::Transpose(%"slice_2") {perm=[0, 1]}
    668 |  # node_Max_668
           %"val_463"<?,?> ⬅️ ::Max(%"getitem")
    669 |  # node_Shape_669
           %"val_464"<?,?> ⬅️ ::Shape(%"val_463") {start=0}
    670 |  # node_Expand_670
           %"val_465"<?,?> ⬅️ ::Expand(%"getitem", %"val_464")
    671 |  # node_Unsqueeze_671
           %"val_466"<?,?> ⬅️ ::Unsqueeze(%"val_465", %"val_42")
    672 |  # node_Concat_672
           %"val_467"<?,?> ⬅️ ::Concat(%"val_466") {axis=-1}
    673 |  # node_GatherND_673
           %"val_468"<?,?> ⬅️ ::GatherND(%"val_462", %"val_467") {batch_dims=0}
    674 |  # node_Transpose_674
           %"index"<FLOAT,[s1,128]> ⬅️ ::Transpose(%"val_468") {perm=[0, 1]}
    675 |  # node_Transpose_675
           %"val_469"<?,?> ⬅️ ::Transpose(%"slice_2") {perm=[0, 1]}
    676 |  # node_Max_676
           %"val_470"<?,?> ⬅️ ::Max(%"getitem_1")
    677 |  # node_Shape_677
           %"val_471"<?,?> ⬅️ ::Shape(%"val_470") {start=0}
    678 |  # node_Expand_678
           %"val_472"<?,?> ⬅️ ::Expand(%"getitem_1", %"val_471")
    679 |  # node_Unsqueeze_679
           %"val_473"<?,?> ⬅️ ::Unsqueeze(%"val_472", %"val_42")
    680 |  # node_Concat_680
           %"val_474"<?,?> ⬅️ ::Concat(%"val_473") {axis=-1}
    681 |  # node_GatherND_681
           %"val_475"<?,?> ⬅️ ::GatherND(%"val_469", %"val_474") {batch_dims=0}
    682 |  # node_Transpose_682
           %"index_1"<FLOAT,[s1,128]> ⬅️ ::Transpose(%"val_475") {perm=[0, 1]}
    683 |  # node_Add_683
           %"add_593"<FLOAT,[s1,128]> ⬅️ ::Add(%"index", %"index_1")
    684 |  # node_Cast_684
           %"scalar_tensor_default"<FLOAT,[]> ⬅️ ::Cast(%"val_13") {to=FLOAT}
    685 |  # node_Div_685
           %"div"<FLOAT,[s1,128]> ⬅️ ::Div(%"add_593", %"scalar_tensor_default")
    686 |  # node_Gemm_686
           %"linear_20"<FLOAT,[s1,128]> ⬅️ ::Gemm(%"div", %"predictors.hidden_layers.lins.0.weight"{...}, %"predictors.hidden_layers.lins.0.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    687 |  # node_Identity_687
           %"clone_21"<FLOAT,[s1,128]> ⬅️ ::Identity(%"linear_20")
    688 |  # node_Add_688
           %"add_606"<FLOAT,[s1,128]> ⬅️ ::Add(%"clone_21", %"div")
    689 |  # node_Gemm_689
           %"cbond"<FLOAT,[s1,1]> ⬅️ ::Gemm(%"add_606", %"predictors.out_layer.weight"{...}, %"predictors.out_layer.bias"{[-0.02880859375]}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    return %"cbond"<FLOAT,[s1,1]>
}

<
    opset_imports={'': 18},
>
def pkg.onnxscript.torch_lib::aten_unbind(
    inputs=(
        %"self"<?,?>
    ),
    attributes={
        dim: INT = 0
    }
    outputs=(
        %"return_val"<?,?>
    ),
) {
    0 |  # n0
         %"split_sizes"<?,?> ⬅️ ::Constant() {value_int=1}
    1 |  # n1
         %"return_val"<?,?> ⬅️ ::SplitToSequence(%"self", %"split_sizes") {keepdims=0, axis=@dim}
    return %"return_val"<?,?>
}

<
    opset_imports={'': 18},
>
def pkg.onnxscript.torch_lib::aten_getitem(
    inputs=(
        %"self"<?,?>,
        %"i"<?,?>
    ),
    outputs=(
        %"return_val"<?,?>
    ),
) {
    0 |  # n0
         %"return_val"<?,?> ⬅️ ::SequenceAt(%"self", %"i")
    return %"return_val"<?,?>
}

<
    opset_imports={'': 18},
>
def pkg.onnxscript.torch_lib.common::Rank(
    inputs=(
        %"input"<?,?>
    ),
    outputs=(
        %"return_val"<?,?>
    ),
) {
    0 |  # n0
         %"tmp"<?,?> ⬅️ ::Shape(%"input")
    1 |  # n1
         %"return_val"<?,?> ⬅️ ::Size(%"tmp")
    return %"return_val"<?,?>
}

<
    opset_imports={'': 18},
>
def pkg.onnxscript.torch_lib.common::IsScalar(
    inputs=(
        %"input"<?,?>
    ),
    outputs=(
        %"return_val"<?,?>
    ),
) {
    0 |  # n0
         %"tmp"<?,?> ⬅️ ::Shape(%"input")
    1 |  # n1
         %"tmp_0"<?,?> ⬅️ ::Size(%"tmp")
    2 |  # n2
         %"tmp_1"<?,?> ⬅️ ::Constant() {value_int=0}
    3 |  # n3
         %"return_val"<?,?> ⬅️ ::Equal(%"tmp_0", %"tmp_1")
    return %"return_val"<?,?>
}
```

## Analysis

PyTorch ONNX Conversion Analysis

## Model Information

The model has 1997697 parameters and 257 buffers (non-trainable parameters).
Number of parameters per dtype:
```python
defaultdict(<class 'int'>, {torch.float32: 1997697})
```
Number of buffers per dtype:
```python
defaultdict(<class 'int'>, {torch.float32: 256, torch.int64: 1})
```

Inputs:
- `x`: `TensorMetadata(shape=torch.Size([s0, 128]), dtype=torch.float32, requires_grad=True, stride=(128, 1), memory_format=torch.contiguous_format, is_quantized=False, qparams={})`
- `padded_xr`: `TensorMetadata(shape=torch.Size([128, 64, 128]), dtype=torch.float32, requires_grad=True, stride=(8192, 128, 1), memory_format=torch.contiguous_format, is_quantized=False, qparams={})`
- `rings_mask`: `TensorMetadata(shape=torch.Size([128, 64]), dtype=torch.bool, requires_grad=False, stride=(64, 1), memory_format=torch.contiguous_format, is_quantized=False, qparams={})`
- `cbond_index`: `TensorMetadata(shape=torch.Size([2, s1]), dtype=torch.int64, requires_grad=False, stride=(1, 2), memory_format=None, is_quantized=False, qparams={})`

Outputs:
- `linear_21`: `TensorMetadata(shape=torch.Size([s1, 1]), dtype=torch.float32, requires_grad=False, stride=(1, 1), memory_format=torch.contiguous_format, is_quantized=False, qparams={})`

The FX graph has 366 nodes in total. Number of FX nodes per op:
- `placeholder`: 131
- `call_function`: 234
- `output`: 1


Of the call_function nodes, the counts of operators used are:

- `aten.view.default`: 59
- `aten.transpose.int`: 30
- `aten.linear.default`: 22
- `aten.clone.default`: 22
- `aten.select.int`: 16
- `aten.add.Tensor`: 12
- `aten.unsqueeze.default`: 10
- `aten.layer_norm.default`: 10
- `aten.squeeze.dim`: 6
- `aten.expand.default`: 5
- `aten.scaled_dot_product_attention.default`: 5
- `aten.permute.default`: 5
- `aten.relu.default`: 5
- `aten.masked_fill.Scalar`: 4
- `<built-in function add>`: 3
- `aten.zeros_like.default`: 2
- `aten.cat.default`: 2
- `aten.zeros.default`: 2
- `<built-in function getitem>`: 2
- `aten.index.Tensor`: 2
- `aten.sym_size.int`: 1
- `aten._unsafe_view.default`: 1
- `aten.abs.default`: 1
- `aten.argmax.default`: 1
- `aten.gather.default`: 1
- `aten.all.dim`: 1
- `aten.unbind.int`: 1
- `aten.slice.Tensor`: 1
- `aten.scalar_tensor.default`: 1
- `aten.div.Tensor`: 1

## ONNX Conversion Information

All operators in the model have registered ONNX decompositions.

## Decomposition comparison

Ops exist only in the ExportedProgram before decomposition: `['aten.contiguous.default', 'aten.dropout.default', 'aten.masked_fill_.Scalar', 'aten.reshape.default', 'aten.unflatten.int']`

Ops exist only in the ExportedProgram after decomposition: `['aten._unsafe_view.default', 'aten.clone.default', 'aten.masked_fill.Scalar', 'aten.scalar_tensor.default']`

