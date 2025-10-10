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
        def forward(self, p_node_processor_x_emb_weight: "f32[120, 128]", p_node_processor_lin_weight: "f32[128, 128]", p_node_processor_lin_bias: "f32[128]", p_node_processor_norm_weight: "f32[128]", p_node_processor_norm_bias: "f32[128]", p_node_processor_graph_convs_0_att: "f32[1, 1, 128]", p_node_processor_graph_convs_0_bias: "f32[128]", p_node_processor_graph_convs_0_lin_l_weight: "f32[128, 128]", p_node_processor_graph_convs_0_lin_l_bias: "f32[128]", p_node_processor_graph_convs_0_lin_r_weight: "f32[128, 128]", p_node_processor_graph_convs_0_lin_r_bias: "f32[128]", p_node_processor_graph_convs_0_lin_edge_weight: "f32[128, 128]", p_node_processor_graph_convs_1_att: "f32[1, 1, 128]", p_node_processor_graph_convs_1_bias: "f32[128]", p_node_processor_graph_convs_1_lin_l_weight: "f32[128, 128]", p_node_processor_graph_convs_1_lin_l_bias: "f32[128]", p_node_processor_graph_convs_1_lin_r_weight: "f32[128, 128]", p_node_processor_graph_convs_1_lin_r_bias: "f32[128]", p_node_processor_graph_convs_1_lin_edge_weight: "f32[128, 128]", p_node_processor_graph_convs_2_att: "f32[1, 1, 128]", p_node_processor_graph_convs_2_bias: "f32[128]", p_node_processor_graph_convs_2_lin_l_weight: "f32[128, 128]", p_node_processor_graph_convs_2_lin_l_bias: "f32[128]", p_node_processor_graph_convs_2_lin_r_weight: "f32[128, 128]", p_node_processor_graph_convs_2_lin_r_bias: "f32[128]", p_node_processor_graph_convs_2_lin_edge_weight: "f32[128, 128]", p_node_processor_graph_convs_3_att: "f32[1, 1, 128]", p_node_processor_graph_convs_3_bias: "f32[128]", p_node_processor_graph_convs_3_lin_l_weight: "f32[128, 128]", p_node_processor_graph_convs_3_lin_l_bias: "f32[128]", p_node_processor_graph_convs_3_lin_r_weight: "f32[128, 128]", p_node_processor_graph_convs_3_lin_r_bias: "f32[128]", p_node_processor_graph_convs_3_lin_edge_weight: "f32[128, 128]", p_node_processor_graph_convs_4_att: "f32[1, 1, 128]", p_node_processor_graph_convs_4_bias: "f32[128]", p_node_processor_graph_convs_4_lin_l_weight: "f32[128, 128]", p_node_processor_graph_convs_4_lin_l_bias: "f32[128]", p_node_processor_graph_convs_4_lin_r_weight: "f32[128, 128]", p_node_processor_graph_convs_4_lin_r_bias: "f32[128]", p_node_processor_graph_convs_4_lin_edge_weight: "f32[128, 128]", p_node_processor_graph_convs_5_att: "f32[1, 1, 128]", p_node_processor_graph_convs_5_bias: "f32[128]", p_node_processor_graph_convs_5_lin_l_weight: "f32[128, 128]", p_node_processor_graph_convs_5_lin_l_bias: "f32[128]", p_node_processor_graph_convs_5_lin_r_weight: "f32[128, 128]", p_node_processor_graph_convs_5_lin_r_bias: "f32[128]", p_node_processor_graph_convs_5_lin_edge_weight: "f32[128, 128]", p_node_processor_graph_norms_0_weight: "f32[128]", p_node_processor_graph_norms_0_bias: "f32[128]", p_node_processor_graph_norms_1_weight: "f32[128]", p_node_processor_graph_norms_1_bias: "f32[128]", p_node_processor_graph_norms_2_weight: "f32[128]", p_node_processor_graph_norms_2_bias: "f32[128]", p_node_processor_graph_norms_3_weight: "f32[128]", p_node_processor_graph_norms_3_bias: "f32[128]", p_node_processor_graph_norms_4_weight: "f32[128]", p_node_processor_graph_norms_4_bias: "f32[128]", b_node_processor_norm_running_mean: "f32[128]", b_node_processor_norm_running_var: "f32[128]", b_node_processor_norm_num_batches_tracked: "i64[]", x: "i32[s0]", edge_index: "i64[2, s1]"):
             # 
            sym_size_int_44: "Sym(s0)" = torch.ops.aten.sym_size.int(x, 0)
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/infer/infer_models/cbond_infer_model.py:285 in forward, code: x = self.x_emb(x.long())
            _to_copy: "i64[s0]" = torch.ops.aten._to_copy.default(x, dtype = torch.int64);  x = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/sparse.py:190 in forward, code: return F.embedding(
            embedding: "f32[s0, 128]" = torch.ops.aten.embedding.default(p_node_processor_x_emb_weight, _to_copy);  p_node_processor_x_emb_weight = _to_copy = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear: "f32[s0, 128]" = torch.ops.aten.linear.default(embedding, p_node_processor_graph_convs_0_lin_l_weight, p_node_processor_graph_convs_0_lin_l_bias);  p_node_processor_graph_convs_0_lin_l_weight = p_node_processor_graph_convs_0_lin_l_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:286 in forward, code: x_l = self.lin_l(x).view(-1, H, C)
            view: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear, [-1, 1, 128]);  linear = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_1: "f32[s0, 128]" = torch.ops.aten.linear.default(embedding, p_node_processor_graph_convs_0_lin_r_weight, p_node_processor_graph_convs_0_lin_r_bias);  p_node_processor_graph_convs_0_lin_r_weight = p_node_processor_graph_convs_0_lin_r_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:290 in forward, code: x_r = self.lin_r(x).view(-1, H, C)
            view_1: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear_1, [-1, 1, 128]);  linear_1 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            select: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 0)
            select_1: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 1)
            ne: "b8[s1]" = torch.ops.aten.ne.Tensor(select, select_1);  select = select_1 = None
            slice_1: "i64[2, s1]" = torch.ops.aten.slice.Tensor(edge_index, 0, 0, 9223372036854775807)
            index: "i64[2, u0]" = torch.ops.aten.index.Tensor(slice_1, [None, ne]);  slice_1 = ne = None
            
             # 
            sym_size_int_46: "Sym(u0)" = torch.ops.aten.sym_size.int(index, 1)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            ge_114: "Sym(u0 >= 0)" = sym_size_int_46 >= 0;  ge_114 = None
            le_6: "Sym(u0 <= 99999)" = sym_size_int_46 <= 99999;  le_6 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:312 in forward, code: edge_index, edge_attr = add_self_loops(
            arange: "i64[s0]" = torch.ops.aten.arange.start(0, sym_size_int_44, device = device(type='cpu'), pin_memory = False)
            view_2: "i64[1, s0]" = torch.ops.aten.view.default(arange, [1, -1]);  arange = None
            repeat: "i64[2, s0]" = torch.ops.aten.repeat.default(view_2, [2, 1]);  view_2 = None
            cat: "i64[2, s0 + u0]" = torch.ops.aten.cat.default([index, repeat], 1);  index = repeat = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:325 in forward, code: alpha = self.edge_updater(edge_index, x=(x_l, x_r),
            select_2: "i64[s0 + u0]" = torch.ops.aten.select.int(cat, 0, 1)
            select_3: "i64[s0 + u0]" = torch.ops.aten.select.int(cat, 0, 0)
            index_select: "f32[s0 + u0, 1, 128]" = torch.ops.aten.index_select.default(view, 0, select_3);  select_3 = None
            index_select_1: "f32[s0 + u0, 1, 128]" = torch.ops.aten.index_select.default(view_1, 0, select_2);  view_1 = None
            add_1225: "Sym(s0 + u0)" = sym_size_int_46 + sym_size_int_44;  sym_size_int_46 = None
            add_53: "f32[s0 + u0, 1, 128]" = torch.ops.aten.add.Tensor(index_select_1, index_select);  index_select_1 = index_select = None
            leaky_relu: "f32[s0 + u0, 1, 128]" = torch.ops.aten.leaky_relu.default(add_53, 0.2);  add_53 = None
            mul_30: "f32[s0 + u0, 1, 128]" = torch.ops.aten.mul.Tensor(leaky_relu, p_node_processor_graph_convs_0_att);  leaky_relu = p_node_processor_graph_convs_0_att = None
            sum_1: "f32[s0 + u0, 1]" = torch.ops.aten.sum.dim_IntList(mul_30, [-1]);  mul_30 = None
            detach: "f32[s0 + u0, 1]" = torch.ops.aten.detach.default(sum_1)
            detach_1: "f32[s0 + u0, 1]" = torch.ops.aten.detach.default(detach);  detach = None
            detach_2: "f32[s0 + u0, 1]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None
            view_3: "i64[s0 + u0, 1]" = torch.ops.aten.view.default(select_2, [-1, 1])
            expand: "i64[s0 + u0, 1]" = torch.ops.aten.expand.default(view_3, [add_1225, 1]);  view_3 = None
            new_zeros: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(detach_2, [sym_size_int_44, 1], pin_memory = False)
            scatter_reduce: "f32[s0, 1]" = torch.ops.aten.scatter_reduce.two(new_zeros, 0, expand, detach_2, 'amax', include_self = False);  new_zeros = expand = detach_2 = None
            index_select_2: "f32[s0 + u0, 1]" = torch.ops.aten.index_select.default(scatter_reduce, 0, select_2);  scatter_reduce = None
            sub_33: "f32[s0 + u0, 1]" = torch.ops.aten.sub.Tensor(sum_1, index_select_2);  sum_1 = index_select_2 = None
            exp: "f32[s0 + u0, 1]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
            view_4: "i64[s0 + u0, 1]" = torch.ops.aten.view.default(select_2, [-1, 1])
            expand_1: "i64[s0 + u0, 1]" = torch.ops.aten.expand.default(view_4, [add_1225, 1]);  view_4 = None
            new_zeros_1: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(exp, [sym_size_int_44, 1], pin_memory = False)
            scatter_add: "f32[s0, 1]" = torch.ops.aten.scatter_add.default(new_zeros_1, 0, expand_1, exp);  new_zeros_1 = expand_1 = None
            add_126: "f32[s0, 1]" = torch.ops.aten.add.Tensor(scatter_add, 1e-16);  scatter_add = None
            index_select_3: "f32[s0 + u0, 1]" = torch.ops.aten.index_select.default(add_126, 0, select_2);  add_126 = select_2 = None
            div: "f32[s0 + u0, 1]" = torch.ops.aten.div.Tensor(exp, index_select_3);  exp = index_select_3 = None
            clone: "f32[s0 + u0, 1]" = torch.ops.aten.clone.default(div);  div = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:329 in forward, code: out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)
            select_4: "i64[s0 + u0]" = torch.ops.aten.select.int(cat, 0, 1)
            select_5: "i64[s0 + u0]" = torch.ops.aten.select.int(cat, 0, 0);  cat = None
            index_select_4: "f32[s0 + u0, 1, 128]" = torch.ops.aten.index_select.default(view, 0, select_5);  view = select_5 = None
            unsqueeze: "f32[s0 + u0, 1, 1]" = torch.ops.aten.unsqueeze.default(clone, -1);  clone = None
            mul_62: "f32[s0 + u0, 1, 128]" = torch.ops.aten.mul.Tensor(index_select_4, unsqueeze);  index_select_4 = unsqueeze = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/aggr/basic.py:22 in forward, code: return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')
            view_5: "i64[s0 + u0, 1, 1]" = torch.ops.aten.view.default(select_4, [-1, 1, 1]);  select_4 = None
            expand_2: "i64[s0 + u0, 1, 128]" = torch.ops.aten.expand.default(view_5, [add_1225, 1, 128]);  view_5 = add_1225 = None
            new_zeros_2: "f32[s0, 1, 128]" = torch.ops.aten.new_zeros.default(mul_62, [sym_size_int_44, 1, 128], pin_memory = False)
            scatter_add_1: "f32[s0, 1, 128]" = torch.ops.aten.scatter_add.default(new_zeros_2, 0, expand_2, mul_62);  new_zeros_2 = expand_2 = mul_62 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:340 in forward, code: out = out + self.bias
            view_7: "f32[s0, 128]" = torch.ops.aten.view.default(scatter_add_1, [-1, 128]);  scatter_add_1 = None
            add_186: "f32[s0, 128]" = torch.ops.aten.add.Tensor(view_7, p_node_processor_graph_convs_0_bias);  view_7 = p_node_processor_graph_convs_0_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:81 in forward, code: x = x - x.mean()
            mean: "f32[]" = torch.ops.aten.mean.default(add_186)
            sub_61: "f32[s0, 128]" = torch.ops.aten.sub.Tensor(add_186, mean);  add_186 = mean = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:82 in forward, code: out = x / (x.std(unbiased=False) + self.eps)
            var: "f32[]" = torch.ops.prims.var.default(sub_61, [0, 1], 0.0)
            sqrt: "f32[]" = torch.ops.aten.sqrt.default(var);  var = None
            add_193: "f32[]" = torch.ops.aten.add.Tensor(sqrt, 1e-05);  sqrt = None
            div_1: "f32[s0, 128]" = torch.ops.aten.div.Tensor(sub_61, add_193);  sub_61 = add_193 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:103 in forward, code: out = out * self.weight + self.bias
            mul_85: "f32[s0, 128]" = torch.ops.aten.mul.Tensor(div_1, p_node_processor_graph_norms_0_weight);  div_1 = p_node_processor_graph_norms_0_weight = None
            add_200: "f32[s0, 128]" = torch.ops.aten.add.Tensor(mul_85, p_node_processor_graph_norms_0_bias);  mul_85 = p_node_processor_graph_norms_0_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:133 in forward, code: return F.relu(input, inplace=self.inplace)
            relu: "f32[s0, 128]" = torch.ops.aten.relu.default(add_200);  add_200 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_1: "f32[s0, 128]" = torch.ops.aten.clone.default(relu);  relu = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_2: "f32[s0, 128]" = torch.ops.aten.linear.default(clone_1, p_node_processor_graph_convs_1_lin_l_weight, p_node_processor_graph_convs_1_lin_l_bias);  p_node_processor_graph_convs_1_lin_l_weight = p_node_processor_graph_convs_1_lin_l_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:286 in forward, code: x_l = self.lin_l(x).view(-1, H, C)
            view_8: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear_2, [-1, 1, 128]);  linear_2 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_3: "f32[s0, 128]" = torch.ops.aten.linear.default(clone_1, p_node_processor_graph_convs_1_lin_r_weight, p_node_processor_graph_convs_1_lin_r_bias);  clone_1 = p_node_processor_graph_convs_1_lin_r_weight = p_node_processor_graph_convs_1_lin_r_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:290 in forward, code: x_r = self.lin_r(x).view(-1, H, C)
            view_9: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear_3, [-1, 1, 128]);  linear_3 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            select_6: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 0)
            select_7: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 1)
            ne_4: "b8[s1]" = torch.ops.aten.ne.Tensor(select_6, select_7);  select_6 = select_7 = None
            slice_2: "i64[2, s1]" = torch.ops.aten.slice.Tensor(edge_index, 0, 0, 9223372036854775807)
            index_1: "i64[2, u1]" = torch.ops.aten.index.Tensor(slice_2, [None, ne_4]);  slice_2 = ne_4 = None
            
             # 
            sym_size_int_47: "Sym(u1)" = torch.ops.aten.sym_size.int(index_1, 1)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            ge_115: "Sym(u1 >= 0)" = sym_size_int_47 >= 0;  ge_115 = None
            le_7: "Sym(u1 <= 99999)" = sym_size_int_47 <= 99999;  le_7 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:312 in forward, code: edge_index, edge_attr = add_self_loops(
            arange_1: "i64[s0]" = torch.ops.aten.arange.start(0, sym_size_int_44, device = device(type='cpu'), pin_memory = False)
            view_10: "i64[1, s0]" = torch.ops.aten.view.default(arange_1, [1, -1]);  arange_1 = None
            repeat_1: "i64[2, s0]" = torch.ops.aten.repeat.default(view_10, [2, 1]);  view_10 = None
            cat_1: "i64[2, s0 + u1]" = torch.ops.aten.cat.default([index_1, repeat_1], 1);  index_1 = repeat_1 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:325 in forward, code: alpha = self.edge_updater(edge_index, x=(x_l, x_r),
            select_8: "i64[s0 + u1]" = torch.ops.aten.select.int(cat_1, 0, 1)
            select_9: "i64[s0 + u1]" = torch.ops.aten.select.int(cat_1, 0, 0)
            index_select_5: "f32[s0 + u1, 1, 128]" = torch.ops.aten.index_select.default(view_8, 0, select_9);  select_9 = None
            index_select_6: "f32[s0 + u1, 1, 128]" = torch.ops.aten.index_select.default(view_9, 0, select_8);  view_9 = None
            add_1226: "Sym(s0 + u1)" = sym_size_int_47 + sym_size_int_44;  sym_size_int_47 = None
            add_258: "f32[s0 + u1, 1, 128]" = torch.ops.aten.add.Tensor(index_select_6, index_select_5);  index_select_6 = index_select_5 = None
            leaky_relu_1: "f32[s0 + u1, 1, 128]" = torch.ops.aten.leaky_relu.default(add_258, 0.2);  add_258 = None
            mul_121: "f32[s0 + u1, 1, 128]" = torch.ops.aten.mul.Tensor(leaky_relu_1, p_node_processor_graph_convs_1_att);  leaky_relu_1 = p_node_processor_graph_convs_1_att = None
            sum_2: "f32[s0 + u1, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [-1]);  mul_121 = None
            detach_3: "f32[s0 + u1, 1]" = torch.ops.aten.detach.default(sum_2)
            detach_4: "f32[s0 + u1, 1]" = torch.ops.aten.detach.default(detach_3);  detach_3 = None
            detach_5: "f32[s0 + u1, 1]" = torch.ops.aten.detach.default(detach_4);  detach_4 = None
            view_11: "i64[s0 + u1, 1]" = torch.ops.aten.view.default(select_8, [-1, 1])
            expand_3: "i64[s0 + u1, 1]" = torch.ops.aten.expand.default(view_11, [add_1226, 1]);  view_11 = None
            new_zeros_3: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(detach_5, [sym_size_int_44, 1], pin_memory = False)
            scatter_reduce_1: "f32[s0, 1]" = torch.ops.aten.scatter_reduce.two(new_zeros_3, 0, expand_3, detach_5, 'amax', include_self = False);  new_zeros_3 = expand_3 = detach_5 = None
            index_select_7: "f32[s0 + u1, 1]" = torch.ops.aten.index_select.default(scatter_reduce_1, 0, select_8);  scatter_reduce_1 = None
            sub_99: "f32[s0 + u1, 1]" = torch.ops.aten.sub.Tensor(sum_2, index_select_7);  sum_2 = index_select_7 = None
            exp_1: "f32[s0 + u1, 1]" = torch.ops.aten.exp.default(sub_99);  sub_99 = None
            view_12: "i64[s0 + u1, 1]" = torch.ops.aten.view.default(select_8, [-1, 1])
            expand_4: "i64[s0 + u1, 1]" = torch.ops.aten.expand.default(view_12, [add_1226, 1]);  view_12 = None
            new_zeros_4: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(exp_1, [sym_size_int_44, 1], pin_memory = False)
            scatter_add_2: "f32[s0, 1]" = torch.ops.aten.scatter_add.default(new_zeros_4, 0, expand_4, exp_1);  new_zeros_4 = expand_4 = None
            add_331: "f32[s0, 1]" = torch.ops.aten.add.Tensor(scatter_add_2, 1e-16);  scatter_add_2 = None
            index_select_8: "f32[s0 + u1, 1]" = torch.ops.aten.index_select.default(add_331, 0, select_8);  add_331 = select_8 = None
            div_2: "f32[s0 + u1, 1]" = torch.ops.aten.div.Tensor(exp_1, index_select_8);  exp_1 = index_select_8 = None
            clone_2: "f32[s0 + u1, 1]" = torch.ops.aten.clone.default(div_2);  div_2 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:329 in forward, code: out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)
            select_10: "i64[s0 + u1]" = torch.ops.aten.select.int(cat_1, 0, 1)
            select_11: "i64[s0 + u1]" = torch.ops.aten.select.int(cat_1, 0, 0);  cat_1 = None
            index_select_9: "f32[s0 + u1, 1, 128]" = torch.ops.aten.index_select.default(view_8, 0, select_11);  view_8 = select_11 = None
            unsqueeze_1: "f32[s0 + u1, 1, 1]" = torch.ops.aten.unsqueeze.default(clone_2, -1);  clone_2 = None
            mul_153: "f32[s0 + u1, 1, 128]" = torch.ops.aten.mul.Tensor(index_select_9, unsqueeze_1);  index_select_9 = unsqueeze_1 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/aggr/basic.py:22 in forward, code: return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')
            view_13: "i64[s0 + u1, 1, 1]" = torch.ops.aten.view.default(select_10, [-1, 1, 1]);  select_10 = None
            expand_5: "i64[s0 + u1, 1, 128]" = torch.ops.aten.expand.default(view_13, [add_1226, 1, 128]);  view_13 = add_1226 = None
            new_zeros_5: "f32[s0, 1, 128]" = torch.ops.aten.new_zeros.default(mul_153, [sym_size_int_44, 1, 128], pin_memory = False)
            scatter_add_3: "f32[s0, 1, 128]" = torch.ops.aten.scatter_add.default(new_zeros_5, 0, expand_5, mul_153);  new_zeros_5 = expand_5 = mul_153 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:340 in forward, code: out = out + self.bias
            view_15: "f32[s0, 128]" = torch.ops.aten.view.default(scatter_add_3, [-1, 128]);  scatter_add_3 = None
            add_391: "f32[s0, 128]" = torch.ops.aten.add.Tensor(view_15, p_node_processor_graph_convs_1_bias);  view_15 = p_node_processor_graph_convs_1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:81 in forward, code: x = x - x.mean()
            mean_1: "f32[]" = torch.ops.aten.mean.default(add_391)
            sub_127: "f32[s0, 128]" = torch.ops.aten.sub.Tensor(add_391, mean_1);  add_391 = mean_1 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:82 in forward, code: out = x / (x.std(unbiased=False) + self.eps)
            var_1: "f32[]" = torch.ops.prims.var.default(sub_127, [0, 1], 0.0)
            sqrt_1: "f32[]" = torch.ops.aten.sqrt.default(var_1);  var_1 = None
            add_398: "f32[]" = torch.ops.aten.add.Tensor(sqrt_1, 1e-05);  sqrt_1 = None
            div_3: "f32[s0, 128]" = torch.ops.aten.div.Tensor(sub_127, add_398);  sub_127 = add_398 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:103 in forward, code: out = out * self.weight + self.bias
            mul_176: "f32[s0, 128]" = torch.ops.aten.mul.Tensor(div_3, p_node_processor_graph_norms_1_weight);  div_3 = p_node_processor_graph_norms_1_weight = None
            add_405: "f32[s0, 128]" = torch.ops.aten.add.Tensor(mul_176, p_node_processor_graph_norms_1_bias);  mul_176 = p_node_processor_graph_norms_1_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:133 in forward, code: return F.relu(input, inplace=self.inplace)
            relu_1: "f32[s0, 128]" = torch.ops.aten.relu.default(add_405);  add_405 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_3: "f32[s0, 128]" = torch.ops.aten.clone.default(relu_1);  relu_1 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_4: "f32[s0, 128]" = torch.ops.aten.linear.default(clone_3, p_node_processor_graph_convs_2_lin_l_weight, p_node_processor_graph_convs_2_lin_l_bias);  p_node_processor_graph_convs_2_lin_l_weight = p_node_processor_graph_convs_2_lin_l_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:286 in forward, code: x_l = self.lin_l(x).view(-1, H, C)
            view_16: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear_4, [-1, 1, 128]);  linear_4 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_5: "f32[s0, 128]" = torch.ops.aten.linear.default(clone_3, p_node_processor_graph_convs_2_lin_r_weight, p_node_processor_graph_convs_2_lin_r_bias);  clone_3 = p_node_processor_graph_convs_2_lin_r_weight = p_node_processor_graph_convs_2_lin_r_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:290 in forward, code: x_r = self.lin_r(x).view(-1, H, C)
            view_17: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear_5, [-1, 1, 128]);  linear_5 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            select_12: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 0)
            select_13: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 1)
            ne_8: "b8[s1]" = torch.ops.aten.ne.Tensor(select_12, select_13);  select_12 = select_13 = None
            slice_3: "i64[2, s1]" = torch.ops.aten.slice.Tensor(edge_index, 0, 0, 9223372036854775807)
            index_2: "i64[2, u2]" = torch.ops.aten.index.Tensor(slice_3, [None, ne_8]);  slice_3 = ne_8 = None
            
             # 
            sym_size_int_48: "Sym(u2)" = torch.ops.aten.sym_size.int(index_2, 1)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            ge_116: "Sym(u2 >= 0)" = sym_size_int_48 >= 0;  ge_116 = None
            le_8: "Sym(u2 <= 99999)" = sym_size_int_48 <= 99999;  le_8 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:312 in forward, code: edge_index, edge_attr = add_self_loops(
            arange_2: "i64[s0]" = torch.ops.aten.arange.start(0, sym_size_int_44, device = device(type='cpu'), pin_memory = False)
            view_18: "i64[1, s0]" = torch.ops.aten.view.default(arange_2, [1, -1]);  arange_2 = None
            repeat_2: "i64[2, s0]" = torch.ops.aten.repeat.default(view_18, [2, 1]);  view_18 = None
            cat_2: "i64[2, s0 + u2]" = torch.ops.aten.cat.default([index_2, repeat_2], 1);  index_2 = repeat_2 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:325 in forward, code: alpha = self.edge_updater(edge_index, x=(x_l, x_r),
            select_14: "i64[s0 + u2]" = torch.ops.aten.select.int(cat_2, 0, 1)
            select_15: "i64[s0 + u2]" = torch.ops.aten.select.int(cat_2, 0, 0)
            index_select_10: "f32[s0 + u2, 1, 128]" = torch.ops.aten.index_select.default(view_16, 0, select_15);  select_15 = None
            index_select_11: "f32[s0 + u2, 1, 128]" = torch.ops.aten.index_select.default(view_17, 0, select_14);  view_17 = None
            add_1227: "Sym(s0 + u2)" = sym_size_int_48 + sym_size_int_44;  sym_size_int_48 = None
            add_463: "f32[s0 + u2, 1, 128]" = torch.ops.aten.add.Tensor(index_select_11, index_select_10);  index_select_11 = index_select_10 = None
            leaky_relu_2: "f32[s0 + u2, 1, 128]" = torch.ops.aten.leaky_relu.default(add_463, 0.2);  add_463 = None
            mul_212: "f32[s0 + u2, 1, 128]" = torch.ops.aten.mul.Tensor(leaky_relu_2, p_node_processor_graph_convs_2_att);  leaky_relu_2 = p_node_processor_graph_convs_2_att = None
            sum_3: "f32[s0 + u2, 1]" = torch.ops.aten.sum.dim_IntList(mul_212, [-1]);  mul_212 = None
            detach_6: "f32[s0 + u2, 1]" = torch.ops.aten.detach.default(sum_3)
            detach_7: "f32[s0 + u2, 1]" = torch.ops.aten.detach.default(detach_6);  detach_6 = None
            detach_8: "f32[s0 + u2, 1]" = torch.ops.aten.detach.default(detach_7);  detach_7 = None
            view_19: "i64[s0 + u2, 1]" = torch.ops.aten.view.default(select_14, [-1, 1])
            expand_6: "i64[s0 + u2, 1]" = torch.ops.aten.expand.default(view_19, [add_1227, 1]);  view_19 = None
            new_zeros_6: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(detach_8, [sym_size_int_44, 1], pin_memory = False)
            scatter_reduce_2: "f32[s0, 1]" = torch.ops.aten.scatter_reduce.two(new_zeros_6, 0, expand_6, detach_8, 'amax', include_self = False);  new_zeros_6 = expand_6 = detach_8 = None
            index_select_12: "f32[s0 + u2, 1]" = torch.ops.aten.index_select.default(scatter_reduce_2, 0, select_14);  scatter_reduce_2 = None
            sub_165: "f32[s0 + u2, 1]" = torch.ops.aten.sub.Tensor(sum_3, index_select_12);  sum_3 = index_select_12 = None
            exp_2: "f32[s0 + u2, 1]" = torch.ops.aten.exp.default(sub_165);  sub_165 = None
            view_20: "i64[s0 + u2, 1]" = torch.ops.aten.view.default(select_14, [-1, 1])
            expand_7: "i64[s0 + u2, 1]" = torch.ops.aten.expand.default(view_20, [add_1227, 1]);  view_20 = None
            new_zeros_7: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(exp_2, [sym_size_int_44, 1], pin_memory = False)
            scatter_add_4: "f32[s0, 1]" = torch.ops.aten.scatter_add.default(new_zeros_7, 0, expand_7, exp_2);  new_zeros_7 = expand_7 = None
            add_536: "f32[s0, 1]" = torch.ops.aten.add.Tensor(scatter_add_4, 1e-16);  scatter_add_4 = None
            index_select_13: "f32[s0 + u2, 1]" = torch.ops.aten.index_select.default(add_536, 0, select_14);  add_536 = select_14 = None
            div_4: "f32[s0 + u2, 1]" = torch.ops.aten.div.Tensor(exp_2, index_select_13);  exp_2 = index_select_13 = None
            clone_4: "f32[s0 + u2, 1]" = torch.ops.aten.clone.default(div_4);  div_4 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:329 in forward, code: out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)
            select_16: "i64[s0 + u2]" = torch.ops.aten.select.int(cat_2, 0, 1)
            select_17: "i64[s0 + u2]" = torch.ops.aten.select.int(cat_2, 0, 0);  cat_2 = None
            index_select_14: "f32[s0 + u2, 1, 128]" = torch.ops.aten.index_select.default(view_16, 0, select_17);  view_16 = select_17 = None
            unsqueeze_2: "f32[s0 + u2, 1, 1]" = torch.ops.aten.unsqueeze.default(clone_4, -1);  clone_4 = None
            mul_244: "f32[s0 + u2, 1, 128]" = torch.ops.aten.mul.Tensor(index_select_14, unsqueeze_2);  index_select_14 = unsqueeze_2 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/aggr/basic.py:22 in forward, code: return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')
            view_21: "i64[s0 + u2, 1, 1]" = torch.ops.aten.view.default(select_16, [-1, 1, 1]);  select_16 = None
            expand_8: "i64[s0 + u2, 1, 128]" = torch.ops.aten.expand.default(view_21, [add_1227, 1, 128]);  view_21 = add_1227 = None
            new_zeros_8: "f32[s0, 1, 128]" = torch.ops.aten.new_zeros.default(mul_244, [sym_size_int_44, 1, 128], pin_memory = False)
            scatter_add_5: "f32[s0, 1, 128]" = torch.ops.aten.scatter_add.default(new_zeros_8, 0, expand_8, mul_244);  new_zeros_8 = expand_8 = mul_244 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:340 in forward, code: out = out + self.bias
            view_23: "f32[s0, 128]" = torch.ops.aten.view.default(scatter_add_5, [-1, 128]);  scatter_add_5 = None
            add_596: "f32[s0, 128]" = torch.ops.aten.add.Tensor(view_23, p_node_processor_graph_convs_2_bias);  view_23 = p_node_processor_graph_convs_2_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:81 in forward, code: x = x - x.mean()
            mean_2: "f32[]" = torch.ops.aten.mean.default(add_596)
            sub_193: "f32[s0, 128]" = torch.ops.aten.sub.Tensor(add_596, mean_2);  add_596 = mean_2 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:82 in forward, code: out = x / (x.std(unbiased=False) + self.eps)
            var_2: "f32[]" = torch.ops.prims.var.default(sub_193, [0, 1], 0.0)
            sqrt_2: "f32[]" = torch.ops.aten.sqrt.default(var_2);  var_2 = None
            add_603: "f32[]" = torch.ops.aten.add.Tensor(sqrt_2, 1e-05);  sqrt_2 = None
            div_5: "f32[s0, 128]" = torch.ops.aten.div.Tensor(sub_193, add_603);  sub_193 = add_603 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:103 in forward, code: out = out * self.weight + self.bias
            mul_267: "f32[s0, 128]" = torch.ops.aten.mul.Tensor(div_5, p_node_processor_graph_norms_2_weight);  div_5 = p_node_processor_graph_norms_2_weight = None
            add_610: "f32[s0, 128]" = torch.ops.aten.add.Tensor(mul_267, p_node_processor_graph_norms_2_bias);  mul_267 = p_node_processor_graph_norms_2_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:133 in forward, code: return F.relu(input, inplace=self.inplace)
            relu_2: "f32[s0, 128]" = torch.ops.aten.relu.default(add_610);  add_610 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_5: "f32[s0, 128]" = torch.ops.aten.clone.default(relu_2);  relu_2 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_6: "f32[s0, 128]" = torch.ops.aten.linear.default(clone_5, p_node_processor_graph_convs_3_lin_l_weight, p_node_processor_graph_convs_3_lin_l_bias);  p_node_processor_graph_convs_3_lin_l_weight = p_node_processor_graph_convs_3_lin_l_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:286 in forward, code: x_l = self.lin_l(x).view(-1, H, C)
            view_24: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear_6, [-1, 1, 128]);  linear_6 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_7: "f32[s0, 128]" = torch.ops.aten.linear.default(clone_5, p_node_processor_graph_convs_3_lin_r_weight, p_node_processor_graph_convs_3_lin_r_bias);  clone_5 = p_node_processor_graph_convs_3_lin_r_weight = p_node_processor_graph_convs_3_lin_r_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:290 in forward, code: x_r = self.lin_r(x).view(-1, H, C)
            view_25: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear_7, [-1, 1, 128]);  linear_7 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            select_18: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 0)
            select_19: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 1)
            ne_12: "b8[s1]" = torch.ops.aten.ne.Tensor(select_18, select_19);  select_18 = select_19 = None
            slice_4: "i64[2, s1]" = torch.ops.aten.slice.Tensor(edge_index, 0, 0, 9223372036854775807)
            index_3: "i64[2, u3]" = torch.ops.aten.index.Tensor(slice_4, [None, ne_12]);  slice_4 = ne_12 = None
            
             # 
            sym_size_int_49: "Sym(u3)" = torch.ops.aten.sym_size.int(index_3, 1)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            ge_117: "Sym(u3 >= 0)" = sym_size_int_49 >= 0;  ge_117 = None
            le_9: "Sym(u3 <= 99999)" = sym_size_int_49 <= 99999;  le_9 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:312 in forward, code: edge_index, edge_attr = add_self_loops(
            arange_3: "i64[s0]" = torch.ops.aten.arange.start(0, sym_size_int_44, device = device(type='cpu'), pin_memory = False)
            view_26: "i64[1, s0]" = torch.ops.aten.view.default(arange_3, [1, -1]);  arange_3 = None
            repeat_3: "i64[2, s0]" = torch.ops.aten.repeat.default(view_26, [2, 1]);  view_26 = None
            cat_3: "i64[2, s0 + u3]" = torch.ops.aten.cat.default([index_3, repeat_3], 1);  index_3 = repeat_3 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:325 in forward, code: alpha = self.edge_updater(edge_index, x=(x_l, x_r),
            select_20: "i64[s0 + u3]" = torch.ops.aten.select.int(cat_3, 0, 1)
            select_21: "i64[s0 + u3]" = torch.ops.aten.select.int(cat_3, 0, 0)
            index_select_15: "f32[s0 + u3, 1, 128]" = torch.ops.aten.index_select.default(view_24, 0, select_21);  select_21 = None
            index_select_16: "f32[s0 + u3, 1, 128]" = torch.ops.aten.index_select.default(view_25, 0, select_20);  view_25 = None
            add_1228: "Sym(s0 + u3)" = sym_size_int_49 + sym_size_int_44;  sym_size_int_49 = None
            add_668: "f32[s0 + u3, 1, 128]" = torch.ops.aten.add.Tensor(index_select_16, index_select_15);  index_select_16 = index_select_15 = None
            leaky_relu_3: "f32[s0 + u3, 1, 128]" = torch.ops.aten.leaky_relu.default(add_668, 0.2);  add_668 = None
            mul_303: "f32[s0 + u3, 1, 128]" = torch.ops.aten.mul.Tensor(leaky_relu_3, p_node_processor_graph_convs_3_att);  leaky_relu_3 = p_node_processor_graph_convs_3_att = None
            sum_4: "f32[s0 + u3, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [-1]);  mul_303 = None
            detach_9: "f32[s0 + u3, 1]" = torch.ops.aten.detach.default(sum_4)
            detach_10: "f32[s0 + u3, 1]" = torch.ops.aten.detach.default(detach_9);  detach_9 = None
            detach_11: "f32[s0 + u3, 1]" = torch.ops.aten.detach.default(detach_10);  detach_10 = None
            view_27: "i64[s0 + u3, 1]" = torch.ops.aten.view.default(select_20, [-1, 1])
            expand_9: "i64[s0 + u3, 1]" = torch.ops.aten.expand.default(view_27, [add_1228, 1]);  view_27 = None
            new_zeros_9: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(detach_11, [sym_size_int_44, 1], pin_memory = False)
            scatter_reduce_3: "f32[s0, 1]" = torch.ops.aten.scatter_reduce.two(new_zeros_9, 0, expand_9, detach_11, 'amax', include_self = False);  new_zeros_9 = expand_9 = detach_11 = None
            index_select_17: "f32[s0 + u3, 1]" = torch.ops.aten.index_select.default(scatter_reduce_3, 0, select_20);  scatter_reduce_3 = None
            sub_231: "f32[s0 + u3, 1]" = torch.ops.aten.sub.Tensor(sum_4, index_select_17);  sum_4 = index_select_17 = None
            exp_3: "f32[s0 + u3, 1]" = torch.ops.aten.exp.default(sub_231);  sub_231 = None
            view_28: "i64[s0 + u3, 1]" = torch.ops.aten.view.default(select_20, [-1, 1])
            expand_10: "i64[s0 + u3, 1]" = torch.ops.aten.expand.default(view_28, [add_1228, 1]);  view_28 = None
            new_zeros_10: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(exp_3, [sym_size_int_44, 1], pin_memory = False)
            scatter_add_6: "f32[s0, 1]" = torch.ops.aten.scatter_add.default(new_zeros_10, 0, expand_10, exp_3);  new_zeros_10 = expand_10 = None
            add_741: "f32[s0, 1]" = torch.ops.aten.add.Tensor(scatter_add_6, 1e-16);  scatter_add_6 = None
            index_select_18: "f32[s0 + u3, 1]" = torch.ops.aten.index_select.default(add_741, 0, select_20);  add_741 = select_20 = None
            div_6: "f32[s0 + u3, 1]" = torch.ops.aten.div.Tensor(exp_3, index_select_18);  exp_3 = index_select_18 = None
            clone_6: "f32[s0 + u3, 1]" = torch.ops.aten.clone.default(div_6);  div_6 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:329 in forward, code: out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)
            select_22: "i64[s0 + u3]" = torch.ops.aten.select.int(cat_3, 0, 1)
            select_23: "i64[s0 + u3]" = torch.ops.aten.select.int(cat_3, 0, 0);  cat_3 = None
            index_select_19: "f32[s0 + u3, 1, 128]" = torch.ops.aten.index_select.default(view_24, 0, select_23);  view_24 = select_23 = None
            unsqueeze_3: "f32[s0 + u3, 1, 1]" = torch.ops.aten.unsqueeze.default(clone_6, -1);  clone_6 = None
            mul_335: "f32[s0 + u3, 1, 128]" = torch.ops.aten.mul.Tensor(index_select_19, unsqueeze_3);  index_select_19 = unsqueeze_3 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/aggr/basic.py:22 in forward, code: return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')
            view_29: "i64[s0 + u3, 1, 1]" = torch.ops.aten.view.default(select_22, [-1, 1, 1]);  select_22 = None
            expand_11: "i64[s0 + u3, 1, 128]" = torch.ops.aten.expand.default(view_29, [add_1228, 1, 128]);  view_29 = add_1228 = None
            new_zeros_11: "f32[s0, 1, 128]" = torch.ops.aten.new_zeros.default(mul_335, [sym_size_int_44, 1, 128], pin_memory = False)
            scatter_add_7: "f32[s0, 1, 128]" = torch.ops.aten.scatter_add.default(new_zeros_11, 0, expand_11, mul_335);  new_zeros_11 = expand_11 = mul_335 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:340 in forward, code: out = out + self.bias
            view_31: "f32[s0, 128]" = torch.ops.aten.view.default(scatter_add_7, [-1, 128]);  scatter_add_7 = None
            add_801: "f32[s0, 128]" = torch.ops.aten.add.Tensor(view_31, p_node_processor_graph_convs_3_bias);  view_31 = p_node_processor_graph_convs_3_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:81 in forward, code: x = x - x.mean()
            mean_3: "f32[]" = torch.ops.aten.mean.default(add_801)
            sub_259: "f32[s0, 128]" = torch.ops.aten.sub.Tensor(add_801, mean_3);  add_801 = mean_3 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:82 in forward, code: out = x / (x.std(unbiased=False) + self.eps)
            var_3: "f32[]" = torch.ops.prims.var.default(sub_259, [0, 1], 0.0)
            sqrt_3: "f32[]" = torch.ops.aten.sqrt.default(var_3);  var_3 = None
            add_808: "f32[]" = torch.ops.aten.add.Tensor(sqrt_3, 1e-05);  sqrt_3 = None
            div_7: "f32[s0, 128]" = torch.ops.aten.div.Tensor(sub_259, add_808);  sub_259 = add_808 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:103 in forward, code: out = out * self.weight + self.bias
            mul_358: "f32[s0, 128]" = torch.ops.aten.mul.Tensor(div_7, p_node_processor_graph_norms_3_weight);  div_7 = p_node_processor_graph_norms_3_weight = None
            add_815: "f32[s0, 128]" = torch.ops.aten.add.Tensor(mul_358, p_node_processor_graph_norms_3_bias);  mul_358 = p_node_processor_graph_norms_3_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:133 in forward, code: return F.relu(input, inplace=self.inplace)
            relu_3: "f32[s0, 128]" = torch.ops.aten.relu.default(add_815);  add_815 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_7: "f32[s0, 128]" = torch.ops.aten.clone.default(relu_3);  relu_3 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_8: "f32[s0, 128]" = torch.ops.aten.linear.default(clone_7, p_node_processor_graph_convs_4_lin_l_weight, p_node_processor_graph_convs_4_lin_l_bias);  p_node_processor_graph_convs_4_lin_l_weight = p_node_processor_graph_convs_4_lin_l_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:286 in forward, code: x_l = self.lin_l(x).view(-1, H, C)
            view_32: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear_8, [-1, 1, 128]);  linear_8 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_9: "f32[s0, 128]" = torch.ops.aten.linear.default(clone_7, p_node_processor_graph_convs_4_lin_r_weight, p_node_processor_graph_convs_4_lin_r_bias);  clone_7 = p_node_processor_graph_convs_4_lin_r_weight = p_node_processor_graph_convs_4_lin_r_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:290 in forward, code: x_r = self.lin_r(x).view(-1, H, C)
            view_33: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear_9, [-1, 1, 128]);  linear_9 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            select_24: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 0)
            select_25: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 1)
            ne_16: "b8[s1]" = torch.ops.aten.ne.Tensor(select_24, select_25);  select_24 = select_25 = None
            slice_5: "i64[2, s1]" = torch.ops.aten.slice.Tensor(edge_index, 0, 0, 9223372036854775807)
            index_4: "i64[2, u4]" = torch.ops.aten.index.Tensor(slice_5, [None, ne_16]);  slice_5 = ne_16 = None
            
             # 
            sym_size_int_50: "Sym(u4)" = torch.ops.aten.sym_size.int(index_4, 1)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            ge_118: "Sym(u4 >= 0)" = sym_size_int_50 >= 0;  ge_118 = None
            le_10: "Sym(u4 <= 99999)" = sym_size_int_50 <= 99999;  le_10 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:312 in forward, code: edge_index, edge_attr = add_self_loops(
            arange_4: "i64[s0]" = torch.ops.aten.arange.start(0, sym_size_int_44, device = device(type='cpu'), pin_memory = False)
            view_34: "i64[1, s0]" = torch.ops.aten.view.default(arange_4, [1, -1]);  arange_4 = None
            repeat_4: "i64[2, s0]" = torch.ops.aten.repeat.default(view_34, [2, 1]);  view_34 = None
            cat_4: "i64[2, s0 + u4]" = torch.ops.aten.cat.default([index_4, repeat_4], 1);  index_4 = repeat_4 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:325 in forward, code: alpha = self.edge_updater(edge_index, x=(x_l, x_r),
            select_26: "i64[s0 + u4]" = torch.ops.aten.select.int(cat_4, 0, 1)
            select_27: "i64[s0 + u4]" = torch.ops.aten.select.int(cat_4, 0, 0)
            index_select_20: "f32[s0 + u4, 1, 128]" = torch.ops.aten.index_select.default(view_32, 0, select_27);  select_27 = None
            index_select_21: "f32[s0 + u4, 1, 128]" = torch.ops.aten.index_select.default(view_33, 0, select_26);  view_33 = None
            add_1229: "Sym(s0 + u4)" = sym_size_int_50 + sym_size_int_44;  sym_size_int_50 = None
            add_873: "f32[s0 + u4, 1, 128]" = torch.ops.aten.add.Tensor(index_select_21, index_select_20);  index_select_21 = index_select_20 = None
            leaky_relu_4: "f32[s0 + u4, 1, 128]" = torch.ops.aten.leaky_relu.default(add_873, 0.2);  add_873 = None
            mul_394: "f32[s0 + u4, 1, 128]" = torch.ops.aten.mul.Tensor(leaky_relu_4, p_node_processor_graph_convs_4_att);  leaky_relu_4 = p_node_processor_graph_convs_4_att = None
            sum_5: "f32[s0 + u4, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [-1]);  mul_394 = None
            detach_12: "f32[s0 + u4, 1]" = torch.ops.aten.detach.default(sum_5)
            detach_13: "f32[s0 + u4, 1]" = torch.ops.aten.detach.default(detach_12);  detach_12 = None
            detach_14: "f32[s0 + u4, 1]" = torch.ops.aten.detach.default(detach_13);  detach_13 = None
            view_35: "i64[s0 + u4, 1]" = torch.ops.aten.view.default(select_26, [-1, 1])
            expand_12: "i64[s0 + u4, 1]" = torch.ops.aten.expand.default(view_35, [add_1229, 1]);  view_35 = None
            new_zeros_12: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(detach_14, [sym_size_int_44, 1], pin_memory = False)
            scatter_reduce_4: "f32[s0, 1]" = torch.ops.aten.scatter_reduce.two(new_zeros_12, 0, expand_12, detach_14, 'amax', include_self = False);  new_zeros_12 = expand_12 = detach_14 = None
            index_select_22: "f32[s0 + u4, 1]" = torch.ops.aten.index_select.default(scatter_reduce_4, 0, select_26);  scatter_reduce_4 = None
            sub_297: "f32[s0 + u4, 1]" = torch.ops.aten.sub.Tensor(sum_5, index_select_22);  sum_5 = index_select_22 = None
            exp_4: "f32[s0 + u4, 1]" = torch.ops.aten.exp.default(sub_297);  sub_297 = None
            view_36: "i64[s0 + u4, 1]" = torch.ops.aten.view.default(select_26, [-1, 1])
            expand_13: "i64[s0 + u4, 1]" = torch.ops.aten.expand.default(view_36, [add_1229, 1]);  view_36 = None
            new_zeros_13: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(exp_4, [sym_size_int_44, 1], pin_memory = False)
            scatter_add_8: "f32[s0, 1]" = torch.ops.aten.scatter_add.default(new_zeros_13, 0, expand_13, exp_4);  new_zeros_13 = expand_13 = None
            add_946: "f32[s0, 1]" = torch.ops.aten.add.Tensor(scatter_add_8, 1e-16);  scatter_add_8 = None
            index_select_23: "f32[s0 + u4, 1]" = torch.ops.aten.index_select.default(add_946, 0, select_26);  add_946 = select_26 = None
            div_8: "f32[s0 + u4, 1]" = torch.ops.aten.div.Tensor(exp_4, index_select_23);  exp_4 = index_select_23 = None
            clone_8: "f32[s0 + u4, 1]" = torch.ops.aten.clone.default(div_8);  div_8 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:329 in forward, code: out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)
            select_28: "i64[s0 + u4]" = torch.ops.aten.select.int(cat_4, 0, 1)
            select_29: "i64[s0 + u4]" = torch.ops.aten.select.int(cat_4, 0, 0);  cat_4 = None
            index_select_24: "f32[s0 + u4, 1, 128]" = torch.ops.aten.index_select.default(view_32, 0, select_29);  view_32 = select_29 = None
            unsqueeze_4: "f32[s0 + u4, 1, 1]" = torch.ops.aten.unsqueeze.default(clone_8, -1);  clone_8 = None
            mul_426: "f32[s0 + u4, 1, 128]" = torch.ops.aten.mul.Tensor(index_select_24, unsqueeze_4);  index_select_24 = unsqueeze_4 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/aggr/basic.py:22 in forward, code: return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')
            view_37: "i64[s0 + u4, 1, 1]" = torch.ops.aten.view.default(select_28, [-1, 1, 1]);  select_28 = None
            expand_14: "i64[s0 + u4, 1, 128]" = torch.ops.aten.expand.default(view_37, [add_1229, 1, 128]);  view_37 = add_1229 = None
            new_zeros_14: "f32[s0, 1, 128]" = torch.ops.aten.new_zeros.default(mul_426, [sym_size_int_44, 1, 128], pin_memory = False)
            scatter_add_9: "f32[s0, 1, 128]" = torch.ops.aten.scatter_add.default(new_zeros_14, 0, expand_14, mul_426);  new_zeros_14 = expand_14 = mul_426 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:340 in forward, code: out = out + self.bias
            view_39: "f32[s0, 128]" = torch.ops.aten.view.default(scatter_add_9, [-1, 128]);  scatter_add_9 = None
            add_1006: "f32[s0, 128]" = torch.ops.aten.add.Tensor(view_39, p_node_processor_graph_convs_4_bias);  view_39 = p_node_processor_graph_convs_4_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:81 in forward, code: x = x - x.mean()
            mean_4: "f32[]" = torch.ops.aten.mean.default(add_1006)
            sub_325: "f32[s0, 128]" = torch.ops.aten.sub.Tensor(add_1006, mean_4);  add_1006 = mean_4 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:82 in forward, code: out = x / (x.std(unbiased=False) + self.eps)
            var_4: "f32[]" = torch.ops.prims.var.default(sub_325, [0, 1], 0.0)
            sqrt_4: "f32[]" = torch.ops.aten.sqrt.default(var_4);  var_4 = None
            add_1013: "f32[]" = torch.ops.aten.add.Tensor(sqrt_4, 1e-05);  sqrt_4 = None
            div_9: "f32[s0, 128]" = torch.ops.aten.div.Tensor(sub_325, add_1013);  sub_325 = add_1013 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/norm/layer_norm.py:103 in forward, code: out = out * self.weight + self.bias
            mul_449: "f32[s0, 128]" = torch.ops.aten.mul.Tensor(div_9, p_node_processor_graph_norms_4_weight);  div_9 = p_node_processor_graph_norms_4_weight = None
            add_1020: "f32[s0, 128]" = torch.ops.aten.add.Tensor(mul_449, p_node_processor_graph_norms_4_bias);  mul_449 = p_node_processor_graph_norms_4_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/activation.py:133 in forward, code: return F.relu(input, inplace=self.inplace)
            relu_4: "f32[s0, 128]" = torch.ops.aten.relu.default(add_1020);  add_1020 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/dropout.py:70 in forward, code: return F.dropout(input, self.p, self.training, self.inplace)
            clone_9: "f32[s0, 128]" = torch.ops.aten.clone.default(relu_4);  relu_4 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_10: "f32[s0, 128]" = torch.ops.aten.linear.default(clone_9, p_node_processor_graph_convs_5_lin_l_weight, p_node_processor_graph_convs_5_lin_l_bias);  p_node_processor_graph_convs_5_lin_l_weight = p_node_processor_graph_convs_5_lin_l_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:286 in forward, code: x_l = self.lin_l(x).view(-1, H, C)
            view_40: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear_10, [-1, 1, 128]);  linear_10 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/dense/linear.py:147 in forward, code: return F.linear(x, self.weight, self.bias)
            linear_11: "f32[s0, 128]" = torch.ops.aten.linear.default(clone_9, p_node_processor_graph_convs_5_lin_r_weight, p_node_processor_graph_convs_5_lin_r_bias);  clone_9 = p_node_processor_graph_convs_5_lin_r_weight = p_node_processor_graph_convs_5_lin_r_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:290 in forward, code: x_r = self.lin_r(x).view(-1, H, C)
            view_41: "f32[s0, 1, 128]" = torch.ops.aten.view.default(linear_11, [-1, 1, 128]);  linear_11 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            select_30: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 0)
            select_31: "i64[s1]" = torch.ops.aten.select.int(edge_index, 0, 1)
            ne_20: "b8[s1]" = torch.ops.aten.ne.Tensor(select_30, select_31);  select_30 = select_31 = None
            slice_6: "i64[2, s1]" = torch.ops.aten.slice.Tensor(edge_index, 0, 0, 9223372036854775807);  edge_index = None
            index_5: "i64[2, u5]" = torch.ops.aten.index.Tensor(slice_6, [None, ne_20]);  slice_6 = ne_20 = None
            
             # 
            sym_size_int_51: "Sym(u5)" = torch.ops.aten.sym_size.int(index_5, 1)
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:310 in forward, code: edge_index, edge_attr = remove_self_loops(
            ge_119: "Sym(u5 >= 0)" = sym_size_int_51 >= 0;  ge_119 = None
            le_11: "Sym(u5 <= 99999)" = sym_size_int_51 <= 99999;  le_11 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:312 in forward, code: edge_index, edge_attr = add_self_loops(
            arange_5: "i64[s0]" = torch.ops.aten.arange.start(0, sym_size_int_44, device = device(type='cpu'), pin_memory = False)
            view_42: "i64[1, s0]" = torch.ops.aten.view.default(arange_5, [1, -1]);  arange_5 = None
            repeat_5: "i64[2, s0]" = torch.ops.aten.repeat.default(view_42, [2, 1]);  view_42 = None
            cat_5: "i64[2, s0 + u5]" = torch.ops.aten.cat.default([index_5, repeat_5], 1);  index_5 = repeat_5 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:325 in forward, code: alpha = self.edge_updater(edge_index, x=(x_l, x_r),
            select_32: "i64[s0 + u5]" = torch.ops.aten.select.int(cat_5, 0, 1)
            select_33: "i64[s0 + u5]" = torch.ops.aten.select.int(cat_5, 0, 0)
            index_select_25: "f32[s0 + u5, 1, 128]" = torch.ops.aten.index_select.default(view_40, 0, select_33);  select_33 = None
            index_select_26: "f32[s0 + u5, 1, 128]" = torch.ops.aten.index_select.default(view_41, 0, select_32);  view_41 = None
            add_1230: "Sym(s0 + u5)" = sym_size_int_51 + sym_size_int_44;  sym_size_int_51 = None
            add_1078: "f32[s0 + u5, 1, 128]" = torch.ops.aten.add.Tensor(index_select_26, index_select_25);  index_select_26 = index_select_25 = None
            leaky_relu_5: "f32[s0 + u5, 1, 128]" = torch.ops.aten.leaky_relu.default(add_1078, 0.2);  add_1078 = None
            mul_485: "f32[s0 + u5, 1, 128]" = torch.ops.aten.mul.Tensor(leaky_relu_5, p_node_processor_graph_convs_5_att);  leaky_relu_5 = p_node_processor_graph_convs_5_att = None
            sum_6: "f32[s0 + u5, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [-1]);  mul_485 = None
            detach_15: "f32[s0 + u5, 1]" = torch.ops.aten.detach.default(sum_6)
            detach_16: "f32[s0 + u5, 1]" = torch.ops.aten.detach.default(detach_15);  detach_15 = None
            detach_17: "f32[s0 + u5, 1]" = torch.ops.aten.detach.default(detach_16);  detach_16 = None
            view_43: "i64[s0 + u5, 1]" = torch.ops.aten.view.default(select_32, [-1, 1])
            expand_15: "i64[s0 + u5, 1]" = torch.ops.aten.expand.default(view_43, [add_1230, 1]);  view_43 = None
            new_zeros_15: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(detach_17, [sym_size_int_44, 1], pin_memory = False)
            scatter_reduce_5: "f32[s0, 1]" = torch.ops.aten.scatter_reduce.two(new_zeros_15, 0, expand_15, detach_17, 'amax', include_self = False);  new_zeros_15 = expand_15 = detach_17 = None
            index_select_27: "f32[s0 + u5, 1]" = torch.ops.aten.index_select.default(scatter_reduce_5, 0, select_32);  scatter_reduce_5 = None
            sub_363: "f32[s0 + u5, 1]" = torch.ops.aten.sub.Tensor(sum_6, index_select_27);  sum_6 = index_select_27 = None
            exp_5: "f32[s0 + u5, 1]" = torch.ops.aten.exp.default(sub_363);  sub_363 = None
            view_44: "i64[s0 + u5, 1]" = torch.ops.aten.view.default(select_32, [-1, 1])
            expand_16: "i64[s0 + u5, 1]" = torch.ops.aten.expand.default(view_44, [add_1230, 1]);  view_44 = None
            new_zeros_16: "f32[s0, 1]" = torch.ops.aten.new_zeros.default(exp_5, [sym_size_int_44, 1], pin_memory = False)
            scatter_add_10: "f32[s0, 1]" = torch.ops.aten.scatter_add.default(new_zeros_16, 0, expand_16, exp_5);  new_zeros_16 = expand_16 = None
            add_1151: "f32[s0, 1]" = torch.ops.aten.add.Tensor(scatter_add_10, 1e-16);  scatter_add_10 = None
            index_select_28: "f32[s0 + u5, 1]" = torch.ops.aten.index_select.default(add_1151, 0, select_32);  add_1151 = select_32 = None
            div_10: "f32[s0 + u5, 1]" = torch.ops.aten.div.Tensor(exp_5, index_select_28);  exp_5 = index_select_28 = None
            clone_10: "f32[s0 + u5, 1]" = torch.ops.aten.clone.default(div_10);  div_10 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:329 in forward, code: out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)
            select_34: "i64[s0 + u5]" = torch.ops.aten.select.int(cat_5, 0, 1)
            select_35: "i64[s0 + u5]" = torch.ops.aten.select.int(cat_5, 0, 0);  cat_5 = None
            index_select_29: "f32[s0 + u5, 1, 128]" = torch.ops.aten.index_select.default(view_40, 0, select_35);  view_40 = select_35 = None
            unsqueeze_5: "f32[s0 + u5, 1, 1]" = torch.ops.aten.unsqueeze.default(clone_10, -1);  clone_10 = None
            mul_517: "f32[s0 + u5, 1, 128]" = torch.ops.aten.mul.Tensor(index_select_29, unsqueeze_5);  index_select_29 = unsqueeze_5 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/aggr/basic.py:22 in forward, code: return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')
            view_45: "i64[s0 + u5, 1, 1]" = torch.ops.aten.view.default(select_34, [-1, 1, 1]);  select_34 = None
            expand_17: "i64[s0 + u5, 1, 128]" = torch.ops.aten.expand.default(view_45, [add_1230, 1, 128]);  view_45 = add_1230 = None
            new_zeros_17: "f32[s0, 1, 128]" = torch.ops.aten.new_zeros.default(mul_517, [sym_size_int_44, 1, 128], pin_memory = False);  sym_size_int_44 = None
            scatter_add_11: "f32[s0, 1, 128]" = torch.ops.aten.scatter_add.default(new_zeros_17, 0, expand_17, mul_517);  new_zeros_17 = expand_17 = mul_517 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:334 in forward, code: out = out.mean(dim=1)
            mean_5: "f32[s0, 128]" = torch.ops.aten.mean.dim(scatter_add_11, [1]);  scatter_add_11 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch_geometric/nn/conv/gatv2_conv.py:340 in forward, code: out = out + self.bias
            add_1211: "f32[s0, 128]" = torch.ops.aten.add.Tensor(mean_5, p_node_processor_graph_convs_5_bias);  mean_5 = p_node_processor_graph_convs_5_bias = None
            
             # File: /mnt/d/hotpot/hotpot/plugins/ComplexFormer/infer/infer_models/cbond_infer_model.py:287 in forward, code: x = self.norm(self.lin(x + xg))
            add_1215: "f32[s0, 128]" = torch.ops.aten.add.Tensor(embedding, add_1211);  embedding = add_1211 = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/linear.py:125 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_12: "f32[s0, 128]" = torch.ops.aten.linear.default(add_1215, p_node_processor_lin_weight, p_node_processor_lin_bias);  add_1215 = p_node_processor_lin_weight = p_node_processor_lin_bias = None
            
             # File: /home/zzy/sw/conda3/envs/hp/lib/python3.9/site-packages/torch/nn/modules/batchnorm.py:193 in forward, code: return F.batch_norm(
            _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(linear_12, p_node_processor_norm_weight, p_node_processor_norm_bias, b_node_processor_norm_running_mean, b_node_processor_norm_running_var, 0.1, 1e-05);  linear_12 = p_node_processor_norm_weight = p_node_processor_norm_bias = b_node_processor_norm_running_mean = b_node_processor_norm_running_var = None
            getitem: "f32[s0, 128]" = _native_batch_norm_legit_no_training[0];  _native_batch_norm_legit_no_training = None
            return (getitem,)
            
Graph signature: ExportGraphSignature(input_specs=[InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_x_emb_weight'), target='node_processor.x_emb.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_lin_weight'), target='node_processor.lin.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_lin_bias'), target='node_processor.lin.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_norm_weight'), target='node_processor.norm.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_norm_bias'), target='node_processor.norm.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_0_att'), target='node_processor.graph.convs.0.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_0_bias'), target='node_processor.graph.convs.0.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_0_lin_l_weight'), target='node_processor.graph.convs.0.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_0_lin_l_bias'), target='node_processor.graph.convs.0.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_0_lin_r_weight'), target='node_processor.graph.convs.0.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_0_lin_r_bias'), target='node_processor.graph.convs.0.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_0_lin_edge_weight'), target='node_processor.graph.convs.0.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_1_att'), target='node_processor.graph.convs.1.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_1_bias'), target='node_processor.graph.convs.1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_1_lin_l_weight'), target='node_processor.graph.convs.1.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_1_lin_l_bias'), target='node_processor.graph.convs.1.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_1_lin_r_weight'), target='node_processor.graph.convs.1.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_1_lin_r_bias'), target='node_processor.graph.convs.1.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_1_lin_edge_weight'), target='node_processor.graph.convs.1.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_2_att'), target='node_processor.graph.convs.2.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_2_bias'), target='node_processor.graph.convs.2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_2_lin_l_weight'), target='node_processor.graph.convs.2.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_2_lin_l_bias'), target='node_processor.graph.convs.2.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_2_lin_r_weight'), target='node_processor.graph.convs.2.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_2_lin_r_bias'), target='node_processor.graph.convs.2.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_2_lin_edge_weight'), target='node_processor.graph.convs.2.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_3_att'), target='node_processor.graph.convs.3.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_3_bias'), target='node_processor.graph.convs.3.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_3_lin_l_weight'), target='node_processor.graph.convs.3.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_3_lin_l_bias'), target='node_processor.graph.convs.3.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_3_lin_r_weight'), target='node_processor.graph.convs.3.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_3_lin_r_bias'), target='node_processor.graph.convs.3.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_3_lin_edge_weight'), target='node_processor.graph.convs.3.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_4_att'), target='node_processor.graph.convs.4.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_4_bias'), target='node_processor.graph.convs.4.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_4_lin_l_weight'), target='node_processor.graph.convs.4.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_4_lin_l_bias'), target='node_processor.graph.convs.4.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_4_lin_r_weight'), target='node_processor.graph.convs.4.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_4_lin_r_bias'), target='node_processor.graph.convs.4.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_4_lin_edge_weight'), target='node_processor.graph.convs.4.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_5_att'), target='node_processor.graph.convs.5.att', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_5_bias'), target='node_processor.graph.convs.5.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_5_lin_l_weight'), target='node_processor.graph.convs.5.lin_l.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_5_lin_l_bias'), target='node_processor.graph.convs.5.lin_l.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_5_lin_r_weight'), target='node_processor.graph.convs.5.lin_r.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_5_lin_r_bias'), target='node_processor.graph.convs.5.lin_r.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_convs_5_lin_edge_weight'), target='node_processor.graph.convs.5.lin_edge.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_norms_0_weight'), target='node_processor.graph.norms.0.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_norms_0_bias'), target='node_processor.graph.norms.0.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_norms_1_weight'), target='node_processor.graph.norms.1.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_norms_1_bias'), target='node_processor.graph.norms.1.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_norms_2_weight'), target='node_processor.graph.norms.2.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_norms_2_bias'), target='node_processor.graph.norms.2.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_norms_3_weight'), target='node_processor.graph.norms.3.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_norms_3_bias'), target='node_processor.graph.norms.3.bias', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_norms_4_weight'), target='node_processor.graph.norms.4.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_node_processor_graph_norms_4_bias'), target='node_processor.graph.norms.4.bias', persistent=None), InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='b_node_processor_norm_running_mean'), target='node_processor.norm.running_mean', persistent=True), InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='b_node_processor_norm_running_var'), target='node_processor.norm.running_var', persistent=True), InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='b_node_processor_norm_num_batches_tracked'), target='node_processor.norm.num_batches_tracked', persistent=True), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='x'), target=None, persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='edge_index'), target=None, persistent=None)], output_specs=[OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='getitem'), target=None)])
Range constraints: {u0: VR[0, 99999], u1: VR[0, 99999], u2: VR[0, 99999], u3: VR[0, 99999], u4: VR[0, 99999], u5: VR[0, 99999], s0: VR[0, 99999], s1: VR[0, 99999]}

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
        %"x"<INT32,[s0]>,
        %"edge_index"<INT64,[2,s1]>
    ),
    outputs=(
        %"xg"<FLOAT,[s0,128]>
    ),
    initializers=(
        %"node_processor.x_emb.weight"<FLOAT,[120,128]>{TorchTensor(...)},
        %"node_processor.lin.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.lin.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.norm.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.norm.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.0.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.0.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.0.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.0.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.0.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.0.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.0.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.1.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.1.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.1.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.1.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.1.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.1.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.1.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.2.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.2.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.2.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.2.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.2.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.2.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.3.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.3.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.3.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.3.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.3.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.3.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.3.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.4.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.4.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.4.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.4.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.4.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.4.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.4.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.5.att"<FLOAT,[1,1,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.5.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.5.lin_l.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.5.lin_l.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.5.lin_r.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.convs.5.lin_r.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.convs.5.lin_edge.weight"<FLOAT,[128,128]>{TorchTensor(...)},
        %"node_processor.graph.norms.0.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.norms.0.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.norms.1.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.norms.1.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.norms.2.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.norms.2.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.norms.3.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.norms.3.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.norms.4.weight"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.graph.norms.4.bias"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.norm.running_mean"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.norm.running_var"<FLOAT,[128]>{TorchTensor(...)},
        %"node_processor.norm.num_batches_tracked"<INT64,[]>{TorchTensor<INT64,[]>(tensor(39792), name='node_processor.norm.num_batches_tracked')}
    ),
) {
      0 |  # node_Shape_0
           %"val_0"<?,?> ⬅️ ::Shape(%"x") {end=1, start=0}
      1 |  # node_Squeeze_1
           %"sym_size_int_44"<INT64,[]> ⬅️ ::Squeeze(%"val_0")
      2 |  # node_Cast_2
           %"_to_copy"<INT64,[s0]> ⬅️ ::Cast(%"x") {to=INT64}
      3 |  # node_Gather_3
           %"embedding"<FLOAT,[s0,128]> ⬅️ ::Gather(%"node_processor.x_emb.weight"{...}, %"_to_copy") {axis=0}
      4 |  # node_Gemm_4
           %"linear"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"embedding", %"node_processor.graph.convs.0.lin_l.weight"{...}, %"node_processor.graph.convs.0.lin_l.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
      5 |  # node_Constant_5
           %"val_1"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[3]>(array([ -1,   1, 128]), name=None)}
      6 |  # node_Cast_6
           %"val_2"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
      7 |  # node_Reshape_7
           %"view"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear", %"val_2") {allowzero=True}
      8 |  # node_Gemm_8
           %"linear_1"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"embedding", %"node_processor.graph.convs.0.lin_r.weight"{...}, %"node_processor.graph.convs.0.lin_r.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
      9 |  # node_Cast_9
           %"val_3"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
     10 |  # node_Reshape_10
           %"view_1"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear_1", %"val_3") {allowzero=True}
     11 |  # node_Constant_11
           %"val_4"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[]>(array(0), name=None)}
     12 |  # node_Gather_12
           %"select"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_4") {axis=0}
     13 |  # node_Constant_13
           %"val_5"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[]>(array(1), name=None)}
     14 |  # node_Gather_14
           %"select_1"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_5") {axis=0}
     15 |  # node_Equal_15
           %"val_6"<?,?> ⬅️ ::Equal(%"select", %"select_1")
     16 |  # node_Not_16
           %"ne"<BOOL,[s1]> ⬅️ ::Not(%"val_6")
     17 |  # node_Cast_17
           %"val_7"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
     18 |  # node_Constant_18
           %"val_8"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
     19 |  # node_Reshape_19
           %"val_9"<?,?> ⬅️ ::Reshape(%"val_7", %"val_8") {allowzero=0}
     20 |  # node_Constant_20
           %"val_10"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[]>(array(9223372036854775807), name=None)}
     21 |  # node_Cast_21
           %"val_11"<?,?> ⬅️ ::Cast(%"val_10") {to=INT64}
     22 |  # node_Constant_22
           %"val_12"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
     23 |  # node_Reshape_23
           %"val_13"<?,?> ⬅️ ::Reshape(%"val_11", %"val_12") {allowzero=0}
     24 |  # node_Cast_24
           %"val_14"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
     25 |  # node_Constant_25
           %"val_15"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
     26 |  # node_Reshape_26
           %"val_16"<?,?> ⬅️ ::Reshape(%"val_14", %"val_15") {allowzero=0}
     27 |  # node_Constant_27
           %"val_17"<?,?> ⬅️ ::Constant() {value_ints=[1]}
     28 |  # node_Slice_28
           %"slice_1"<INT64,[2,s1]> ⬅️ ::Slice(%"edge_index", %"val_9", %"val_13", %"val_16", %"val_17")
     29 |  # node_NonZero_29
           %"val_18"<?,?> ⬅️ ::NonZero(%"ne")
     30 |  # node_Transpose_30
           %"val_19"<?,?> ⬅️ ::Transpose(%"val_18") {perm=[1, 0]}
     31 |  # node_Constant_31
           %"val_20"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([1]), name=None)}
     32 |  # node_Squeeze_32
           %"val_21"<?,?> ⬅️ ::Squeeze(%"val_19", %"val_20")
     33 |  # node_Transpose_33
           %"val_22"<?,?> ⬅️ ::Transpose(%"slice_1") {perm=[1, 0]}
     34 |  # node_Max_34
           %"val_23"<?,?> ⬅️ ::Max(%"val_21")
     35 |  # node_Shape_35
           %"val_24"<?,?> ⬅️ ::Shape(%"val_23") {start=0}
     36 |  # node_Expand_36
           %"val_25"<?,?> ⬅️ ::Expand(%"val_21", %"val_24")
     37 |  # node_Constant_37
           %"val_26"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
     38 |  # node_Unsqueeze_38
           %"val_27"<?,?> ⬅️ ::Unsqueeze(%"val_25", %"val_26")
     39 |  # node_Concat_39
           %"val_28"<?,?> ⬅️ ::Concat(%"val_27") {axis=-1}
     40 |  # node_GatherND_40
           %"val_29"<?,?> ⬅️ ::GatherND(%"val_22", %"val_28") {batch_dims=0}
     41 |  # node_Transpose_41
           %"index"<INT64,[2,u0]> ⬅️ ::Transpose(%"val_29") {perm=[1, 0]}
     42 |  # node_Shape_42
           %"val_30"<?,?> ⬅️ ::Shape(%"index") {end=2, start=1}
     43 |  # node_Squeeze_43
           %"sym_size_int_46"<INT64,[]> ⬅️ ::Squeeze(%"val_30")
     44 |  # node_GreaterOrEqual_44
           %"ge_114"<BOOL,[]> ⬅️ ::GreaterOrEqual(%"sym_size_int_46", %"val_4")
     45 |  # node_Constant_45
           %"val_31"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[]>(array(99999), name=None)}
     46 |  # node_LessOrEqual_46
           %"le_6"<BOOL,[]> ⬅️ ::LessOrEqual(%"sym_size_int_46", %"val_31")
     47 |  # node_Constant_47
           %"val_32"<?,?> ⬅️ ::Constant() {value=Tensor<FLOAT,[]>(array(1., dtype=float32), name=None)}
     48 |  # node_CastLike_48
           %"val_33"<?,?> ⬅️ ::CastLike(%"val_32", %"sym_size_int_44")
     49 |  # node_Range_49
           %"arange"<INT64,[s0]> ⬅️ ::Range(%"val_4", %"sym_size_int_44", %"val_33")
     50 |  # node_Constant_50
           %"val_34"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[2]>(array([ 1, -1]), name=None)}
     51 |  # node_Cast_51
           %"val_35"<?,?> ⬅️ ::Cast(%"val_34") {to=INT64}
     52 |  # node_Reshape_52
           %"view_2"<INT64,[1,s0]> ⬅️ ::Reshape(%"arange", %"val_35") {allowzero=True}
     53 |  # node_Constant_53
           %"val_36"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[2]>(array([1, 1]), name=None)}
     54 |  # node_Expand_54
           %"val_37"<?,?> ⬅️ ::Expand(%"view_2", %"val_36")
     55 |  # node_Constant_55
           %"val_38"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[2]>(array([2, 1]), name=None)}
     56 |  # node_Tile_56
           %"repeat"<INT64,[2,s0]> ⬅️ ::Tile(%"val_37", %"val_38")
     57 |  # node_Concat_57
           %"cat"<INT64,[2,s0 + u0]> ⬅️ ::Concat(%"index", %"repeat") {axis=1}
     58 |  # node_Gather_58
           %"select_2"<INT64,[s0 + u0]> ⬅️ ::Gather(%"cat", %"val_5") {axis=0}
     59 |  # node_Gather_59
           %"select_3"<INT64,[s0 + u0]> ⬅️ ::Gather(%"cat", %"val_4") {axis=0}
     60 |  # node_Constant_60
           %"val_39"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
     61 |  # node_Reshape_61
           %"val_40"<?,?> ⬅️ ::Reshape(%"select_3", %"val_39") {allowzero=0}
     62 |  # node_Cast_62
           %"val_41"<?,?> ⬅️ ::Cast(%"val_40") {to=INT64}
     63 |  # node_Gather_63
           %"index_select"<FLOAT,[s0 + u0,1,128]> ⬅️ ::Gather(%"view", %"val_41") {axis=0}
     64 |  # node_Constant_64
           %"val_42"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
     65 |  # node_Reshape_65
           %"val_43"<?,?> ⬅️ ::Reshape(%"select_2", %"val_42") {allowzero=0}
     66 |  # node_Cast_66
           %"val_44"<?,?> ⬅️ ::Cast(%"val_43") {to=INT64}
     67 |  # node_Gather_67
           %"index_select_1"<FLOAT,[s0 + u0,1,128]> ⬅️ ::Gather(%"view_1", %"val_44") {axis=0}
     68 |  # node_Add_68
           %"add_1225"<INT64,[]> ⬅️ ::Add(%"sym_size_int_46", %"sym_size_int_44")
     69 |  # node_Add_69
           %"add_53"<FLOAT,[s0 + u0,1,128]> ⬅️ ::Add(%"index_select_1", %"index_select")
     70 |  # node_LeakyRelu_70
           %"leaky_relu"<FLOAT,[s0 + u0,1,128]> ⬅️ ::LeakyRelu(%"add_53") {alpha=0.2}
     71 |  # node_Mul_71
           %"mul_30"<FLOAT,[s0 + u0,1,128]> ⬅️ ::Mul(%"leaky_relu", %"node_processor.graph.convs.0.att"{...})
     72 |  # node_Constant_72
           %"val_45"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
     73 |  # node_Reshape_73
           %"val_46"<?,?> ⬅️ ::Reshape(%"val_26", %"val_45") {allowzero=0}
     74 |  # node_Cast_74
           %"val_47"<?,?> ⬅️ ::Cast(%"val_46") {to=INT64}
     75 |  # node_ReduceSum_75
           %"sum_1"<FLOAT,[s0 + u0,1]> ⬅️ ::ReduceSum(%"mul_30", %"val_47") {noop_with_empty_axes=0, keepdims=False}
     76 |  # node_Identity_76
           %"detach"<FLOAT,[s0 + u0,1]> ⬅️ ::Identity(%"sum_1")
     77 |  # node_Identity_77
           %"detach_1"<FLOAT,[s0 + u0,1]> ⬅️ ::Identity(%"detach")
     78 |  # node_Identity_78
           %"detach_2"<FLOAT,[s0 + u0,1]> ⬅️ ::Identity(%"detach_1")
     79 |  # node_Constant_79
           %"val_48"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[2]>(array([-1,  1]), name=None)}
     80 |  # node_Cast_80
           %"val_49"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
     81 |  # node_Reshape_81
           %"view_3"<INT64,[s0 + u0,1]> ⬅️ ::Reshape(%"select_2", %"val_49") {allowzero=True}
     82 |  # node_Constant_82
           %"val_50"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
     83 |  # node_Reshape_83
           %"val_51"<?,?> ⬅️ ::Reshape(%"add_1225", %"val_50") {allowzero=0}
     84 |  # node_Concat_84
           %"val_52"<?,?> ⬅️ ::Concat(%"val_51", %"val_20") {axis=0}
     85 |  # node_Cast_85
           %"val_53"<?,?> ⬅️ ::Cast(%"val_52") {to=INT64}
     86 |  # node_Abs_86
           %"val_54"<?,?> ⬅️ ::Abs(%"val_53")
     87 |  # node_Expand_87
           %"expand"<INT64,[s0 + u0,1]> ⬅️ ::Expand(%"view_3", %"val_54")
     88 |  # node_Constant_88
           %"val_55"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
     89 |  # node_Reshape_89
           %"val_56"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_55") {allowzero=0}
     90 |  # node_Concat_90
           %"val_57"<?,?> ⬅️ ::Concat(%"val_56", %"val_20") {axis=0}
     91 |  # node_ConstantOfShape_91
           %"val_58"<?,?> ⬅️ ::ConstantOfShape(%"val_57")
     92 |  # node_CastLike_92
           %"new_zeros"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_58", %"detach_2")
     93 |  # node_Shape_93
           %"val_59"<?,?> ⬅️ ::Shape(%"detach_2") {start=0}
     94 |  # node_ConstantOfShape_94
           %"val_60"<?,?> ⬅️ ::ConstantOfShape(%"val_59") {value=Tensor<FLOAT,[1]>(array([-3.4028235e+38], dtype=float32), name=None)}
     95 |  # node_ScatterElements_95
           %"val_61"<?,?> ⬅️ ::ScatterElements(%"new_zeros", %"expand", %"val_60") {reduction=min, axis=0}
     96 |  # node_ScatterElements_96
           %"scatter_reduce"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"val_61", %"expand", %"detach_2") {reduction=max, axis=0}
     97 |  # node_Constant_97
           %"val_62"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
     98 |  # node_Reshape_98
           %"val_63"<?,?> ⬅️ ::Reshape(%"select_2", %"val_62") {allowzero=0}
     99 |  # node_Cast_99
           %"val_64"<?,?> ⬅️ ::Cast(%"val_63") {to=INT64}
    100 |  # node_Gather_100
           %"index_select_2"<FLOAT,[s0 + u0,1]> ⬅️ ::Gather(%"scatter_reduce", %"val_64") {axis=0}
    101 |  # node_Sub_101
           %"sub_33"<FLOAT,[s0 + u0,1]> ⬅️ ::Sub(%"sum_1", %"index_select_2")
    102 |  # node_Exp_102
           %"exp"<FLOAT,[s0 + u0,1]> ⬅️ ::Exp(%"sub_33")
    103 |  # node_Cast_103
           %"val_65"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
    104 |  # node_Reshape_104
           %"view_4"<INT64,[s0 + u0,1]> ⬅️ ::Reshape(%"select_2", %"val_65") {allowzero=True}
    105 |  # node_Constant_105
           %"val_66"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    106 |  # node_Reshape_106
           %"val_67"<?,?> ⬅️ ::Reshape(%"add_1225", %"val_66") {allowzero=0}
    107 |  # node_Concat_107
           %"val_68"<?,?> ⬅️ ::Concat(%"val_67", %"val_20") {axis=0}
    108 |  # node_Cast_108
           %"val_69"<?,?> ⬅️ ::Cast(%"val_68") {to=INT64}
    109 |  # node_Abs_109
           %"val_70"<?,?> ⬅️ ::Abs(%"val_69")
    110 |  # node_Expand_110
           %"expand_1"<INT64,[s0 + u0,1]> ⬅️ ::Expand(%"view_4", %"val_70")
    111 |  # node_Constant_111
           %"val_71"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    112 |  # node_Reshape_112
           %"val_72"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_71") {allowzero=0}
    113 |  # node_Concat_113
           %"val_73"<?,?> ⬅️ ::Concat(%"val_72", %"val_20") {axis=0}
    114 |  # node_ConstantOfShape_114
           %"val_74"<?,?> ⬅️ ::ConstantOfShape(%"val_73")
    115 |  # node_CastLike_115
           %"new_zeros_1"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_74", %"exp")
    116 |  # node_ScatterElements_116
           %"scatter_add"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"new_zeros_1", %"expand_1", %"exp") {reduction=add, axis=0}
    117 |  # node_Constant_117
           %"val_75"<?,?> ⬅️ ::Constant() {value=Tensor<FLOAT,[]>(array(1.e-16, dtype=float32), name=None)}
    118 |  # node_Add_118
           %"add_126"<FLOAT,[s0,1]> ⬅️ ::Add(%"scatter_add", %"val_75")
    119 |  # node_Constant_119
           %"val_76"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    120 |  # node_Reshape_120
           %"val_77"<?,?> ⬅️ ::Reshape(%"select_2", %"val_76") {allowzero=0}
    121 |  # node_Cast_121
           %"val_78"<?,?> ⬅️ ::Cast(%"val_77") {to=INT64}
    122 |  # node_Gather_122
           %"index_select_3"<FLOAT,[s0 + u0,1]> ⬅️ ::Gather(%"add_126", %"val_78") {axis=0}
    123 |  # node_Div_123
           %"div"<FLOAT,[s0 + u0,1]> ⬅️ ::Div(%"exp", %"index_select_3")
    124 |  # node_Identity_124
           %"clone"<FLOAT,[s0 + u0,1]> ⬅️ ::Identity(%"div")
    125 |  # node_Gather_125
           %"select_4"<INT64,[s0 + u0]> ⬅️ ::Gather(%"cat", %"val_5") {axis=0}
    126 |  # node_Gather_126
           %"select_5"<INT64,[s0 + u0]> ⬅️ ::Gather(%"cat", %"val_4") {axis=0}
    127 |  # node_Constant_127
           %"val_79"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    128 |  # node_Reshape_128
           %"val_80"<?,?> ⬅️ ::Reshape(%"select_5", %"val_79") {allowzero=0}
    129 |  # node_Cast_129
           %"val_81"<?,?> ⬅️ ::Cast(%"val_80") {to=INT64}
    130 |  # node_Gather_130
           %"index_select_4"<FLOAT,[s0 + u0,1,128]> ⬅️ ::Gather(%"view", %"val_81") {axis=0}
    131 |  # node_Unsqueeze_131
           %"unsqueeze"<FLOAT,[s0 + u0,1,1]> ⬅️ ::Unsqueeze(%"clone", %"val_26")
    132 |  # node_Mul_132
           %"mul_62"<FLOAT,[s0 + u0,1,128]> ⬅️ ::Mul(%"index_select_4", %"unsqueeze")
    133 |  # node_Constant_133
           %"val_82"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[3]>(array([-1,  1,  1]), name=None)}
    134 |  # node_Cast_134
           %"val_83"<?,?> ⬅️ ::Cast(%"val_82") {to=INT64}
    135 |  # node_Reshape_135
           %"view_5"<INT64,[s0 + u0,1,1]> ⬅️ ::Reshape(%"select_4", %"val_83") {allowzero=True}
    136 |  # node_Constant_136
           %"val_84"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    137 |  # node_Reshape_137
           %"val_85"<?,?> ⬅️ ::Reshape(%"add_1225", %"val_84") {allowzero=0}
    138 |  # node_Constant_138
           %"val_86"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([128]), name=None)}
    139 |  # node_Concat_139
           %"val_87"<?,?> ⬅️ ::Concat(%"val_85", %"val_20", %"val_86") {axis=0}
    140 |  # node_Cast_140
           %"val_88"<?,?> ⬅️ ::Cast(%"val_87") {to=INT64}
    141 |  # node_Abs_141
           %"val_89"<?,?> ⬅️ ::Abs(%"val_88")
    142 |  # node_Expand_142
           %"expand_2"<INT64,[s0 + u0,1,128]> ⬅️ ::Expand(%"view_5", %"val_89")
    143 |  # node_Constant_143
           %"val_90"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    144 |  # node_Reshape_144
           %"val_91"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_90") {allowzero=0}
    145 |  # node_Concat_145
           %"val_92"<?,?> ⬅️ ::Concat(%"val_91", %"val_20", %"val_86") {axis=0}
    146 |  # node_ConstantOfShape_146
           %"val_93"<?,?> ⬅️ ::ConstantOfShape(%"val_92")
    147 |  # node_CastLike_147
           %"new_zeros_2"<FLOAT,[s0,1,128]> ⬅️ ::CastLike(%"val_93", %"mul_62")
    148 |  # node_ScatterElements_148
           %"scatter_add_1"<FLOAT,[s0,1,128]> ⬅️ ::ScatterElements(%"new_zeros_2", %"expand_2", %"mul_62") {reduction=add, axis=0}
    149 |  # node_Constant_149
           %"val_94"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[2]>(array([ -1, 128]), name=None)}
    150 |  # node_Cast_150
           %"val_95"<?,?> ⬅️ ::Cast(%"val_94") {to=INT64}
    151 |  # node_Reshape_151
           %"view_7"<FLOAT,[s0,128]> ⬅️ ::Reshape(%"scatter_add_1", %"val_95") {allowzero=True}
    152 |  # node_Add_152
           %"add_186"<FLOAT,[s0,128]> ⬅️ ::Add(%"view_7", %"node_processor.graph.convs.0.bias"{...})
    153 |  # node_aten_mean_153
           %"mean"<FLOAT,[]> ⬅️ pkg.onnxscript.torch_lib::aten_mean(%"add_186")
    154 |  # node_Sub_154
           %"sub_61"<FLOAT,[s0,128]> ⬅️ ::Sub(%"add_186", %"mean")
    155 |  # node_Constant_155
           %"val_96"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[2]>(array([0, 1]), name=None)}
    156 |  # node_ReduceMean_156
           %"val_97"<?,?> ⬅️ ::ReduceMean(%"sub_61", %"val_96") {noop_with_empty_axes=0, keepdims=True}
    157 |  # node_Sub_157
           %"val_98"<?,?> ⬅️ ::Sub(%"sub_61", %"val_97")
    158 |  # node_Mul_158
           %"val_99"<?,?> ⬅️ ::Mul(%"val_98", %"val_98")
    159 |  # node_ReduceMean_159
           %"var"<FLOAT,[]> ⬅️ ::ReduceMean(%"val_99", %"val_96") {noop_with_empty_axes=0, keepdims=False}
    160 |  # node_Sqrt_160
           %"sqrt"<FLOAT,[]> ⬅️ ::Sqrt(%"var")
    161 |  # node_Constant_161
           %"val_100"<?,?> ⬅️ ::Constant() {value=Tensor<FLOAT,[]>(array(1.e-05, dtype=float32), name=None)}
    162 |  # node_Add_162
           %"add_193"<FLOAT,[]> ⬅️ ::Add(%"sqrt", %"val_100")
    163 |  # node_Div_163
           %"div_1"<FLOAT,[s0,128]> ⬅️ ::Div(%"sub_61", %"add_193")
    164 |  # node_Mul_164
           %"mul_85"<FLOAT,[s0,128]> ⬅️ ::Mul(%"div_1", %"node_processor.graph.norms.0.weight"{...})
    165 |  # node_Add_165
           %"add_200"<FLOAT,[s0,128]> ⬅️ ::Add(%"mul_85", %"node_processor.graph.norms.0.bias"{...})
    166 |  # node_Relu_166
           %"relu"<FLOAT,[s0,128]> ⬅️ ::Relu(%"add_200")
    167 |  # node_Identity_167
           %"clone_1"<FLOAT,[s0,128]> ⬅️ ::Identity(%"relu")
    168 |  # node_Gemm_168
           %"linear_2"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"clone_1", %"node_processor.graph.convs.1.lin_l.weight"{...}, %"node_processor.graph.convs.1.lin_l.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    169 |  # node_Cast_169
           %"val_101"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
    170 |  # node_Reshape_170
           %"view_8"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear_2", %"val_101") {allowzero=True}
    171 |  # node_Gemm_171
           %"linear_3"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"clone_1", %"node_processor.graph.convs.1.lin_r.weight"{...}, %"node_processor.graph.convs.1.lin_r.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    172 |  # node_Cast_172
           %"val_102"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
    173 |  # node_Reshape_173
           %"view_9"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear_3", %"val_102") {allowzero=True}
    174 |  # node_Gather_174
           %"select_6"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_4") {axis=0}
    175 |  # node_Gather_175
           %"select_7"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_5") {axis=0}
    176 |  # node_Equal_176
           %"val_103"<?,?> ⬅️ ::Equal(%"select_6", %"select_7")
    177 |  # node_Not_177
           %"ne_4"<BOOL,[s1]> ⬅️ ::Not(%"val_103")
    178 |  # node_Cast_178
           %"val_104"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
    179 |  # node_Constant_179
           %"val_105"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    180 |  # node_Reshape_180
           %"val_106"<?,?> ⬅️ ::Reshape(%"val_104", %"val_105") {allowzero=0}
    181 |  # node_Cast_181
           %"val_107"<?,?> ⬅️ ::Cast(%"val_10") {to=INT64}
    182 |  # node_Constant_182
           %"val_108"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    183 |  # node_Reshape_183
           %"val_109"<?,?> ⬅️ ::Reshape(%"val_107", %"val_108") {allowzero=0}
    184 |  # node_Cast_184
           %"val_110"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
    185 |  # node_Constant_185
           %"val_111"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    186 |  # node_Reshape_186
           %"val_112"<?,?> ⬅️ ::Reshape(%"val_110", %"val_111") {allowzero=0}
    187 |  # node_Constant_187
           %"val_113"<?,?> ⬅️ ::Constant() {value_ints=[1]}
    188 |  # node_Slice_188
           %"slice_2"<INT64,[2,s1]> ⬅️ ::Slice(%"edge_index", %"val_106", %"val_109", %"val_112", %"val_113")
    189 |  # node_NonZero_189
           %"val_114"<?,?> ⬅️ ::NonZero(%"ne_4")
    190 |  # node_Transpose_190
           %"val_115"<?,?> ⬅️ ::Transpose(%"val_114") {perm=[1, 0]}
    191 |  # node_Squeeze_191
           %"val_116"<?,?> ⬅️ ::Squeeze(%"val_115", %"val_20")
    192 |  # node_Transpose_192
           %"val_117"<?,?> ⬅️ ::Transpose(%"slice_2") {perm=[1, 0]}
    193 |  # node_Max_193
           %"val_118"<?,?> ⬅️ ::Max(%"val_116")
    194 |  # node_Shape_194
           %"val_119"<?,?> ⬅️ ::Shape(%"val_118") {start=0}
    195 |  # node_Expand_195
           %"val_120"<?,?> ⬅️ ::Expand(%"val_116", %"val_119")
    196 |  # node_Unsqueeze_196
           %"val_121"<?,?> ⬅️ ::Unsqueeze(%"val_120", %"val_26")
    197 |  # node_Concat_197
           %"val_122"<?,?> ⬅️ ::Concat(%"val_121") {axis=-1}
    198 |  # node_GatherND_198
           %"val_123"<?,?> ⬅️ ::GatherND(%"val_117", %"val_122") {batch_dims=0}
    199 |  # node_Transpose_199
           %"index_1"<INT64,[2,u1]> ⬅️ ::Transpose(%"val_123") {perm=[1, 0]}
    200 |  # node_Shape_200
           %"val_124"<?,?> ⬅️ ::Shape(%"index_1") {end=2, start=1}
    201 |  # node_Squeeze_201
           %"sym_size_int_47"<INT64,[]> ⬅️ ::Squeeze(%"val_124")
    202 |  # node_GreaterOrEqual_202
           %"ge_115"<BOOL,[]> ⬅️ ::GreaterOrEqual(%"sym_size_int_47", %"val_4")
    203 |  # node_LessOrEqual_203
           %"le_7"<BOOL,[]> ⬅️ ::LessOrEqual(%"sym_size_int_47", %"val_31")
    204 |  # node_CastLike_204
           %"val_125"<?,?> ⬅️ ::CastLike(%"val_32", %"sym_size_int_44")
    205 |  # node_Range_205
           %"arange_1"<INT64,[s0]> ⬅️ ::Range(%"val_4", %"sym_size_int_44", %"val_125")
    206 |  # node_Cast_206
           %"val_126"<?,?> ⬅️ ::Cast(%"val_34") {to=INT64}
    207 |  # node_Reshape_207
           %"view_10"<INT64,[1,s0]> ⬅️ ::Reshape(%"arange_1", %"val_126") {allowzero=True}
    208 |  # node_Expand_208
           %"val_127"<?,?> ⬅️ ::Expand(%"view_10", %"val_36")
    209 |  # node_Tile_209
           %"repeat_1"<INT64,[2,s0]> ⬅️ ::Tile(%"val_127", %"val_38")
    210 |  # node_Concat_210
           %"cat_1"<INT64,[2,s0 + u1]> ⬅️ ::Concat(%"index_1", %"repeat_1") {axis=1}
    211 |  # node_Gather_211
           %"select_8"<INT64,[s0 + u1]> ⬅️ ::Gather(%"cat_1", %"val_5") {axis=0}
    212 |  # node_Gather_212
           %"select_9"<INT64,[s0 + u1]> ⬅️ ::Gather(%"cat_1", %"val_4") {axis=0}
    213 |  # node_Constant_213
           %"val_128"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    214 |  # node_Reshape_214
           %"val_129"<?,?> ⬅️ ::Reshape(%"select_9", %"val_128") {allowzero=0}
    215 |  # node_Cast_215
           %"val_130"<?,?> ⬅️ ::Cast(%"val_129") {to=INT64}
    216 |  # node_Gather_216
           %"index_select_5"<FLOAT,[s0 + u1,1,128]> ⬅️ ::Gather(%"view_8", %"val_130") {axis=0}
    217 |  # node_Constant_217
           %"val_131"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    218 |  # node_Reshape_218
           %"val_132"<?,?> ⬅️ ::Reshape(%"select_8", %"val_131") {allowzero=0}
    219 |  # node_Cast_219
           %"val_133"<?,?> ⬅️ ::Cast(%"val_132") {to=INT64}
    220 |  # node_Gather_220
           %"index_select_6"<FLOAT,[s0 + u1,1,128]> ⬅️ ::Gather(%"view_9", %"val_133") {axis=0}
    221 |  # node_Add_221
           %"add_1226"<INT64,[]> ⬅️ ::Add(%"sym_size_int_47", %"sym_size_int_44")
    222 |  # node_Add_222
           %"add_258"<FLOAT,[s0 + u1,1,128]> ⬅️ ::Add(%"index_select_6", %"index_select_5")
    223 |  # node_LeakyRelu_223
           %"leaky_relu_1"<FLOAT,[s0 + u1,1,128]> ⬅️ ::LeakyRelu(%"add_258") {alpha=0.2}
    224 |  # node_Mul_224
           %"mul_121"<FLOAT,[s0 + u1,1,128]> ⬅️ ::Mul(%"leaky_relu_1", %"node_processor.graph.convs.1.att"{...})
    225 |  # node_Constant_225
           %"val_134"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    226 |  # node_Reshape_226
           %"val_135"<?,?> ⬅️ ::Reshape(%"val_26", %"val_134") {allowzero=0}
    227 |  # node_Cast_227
           %"val_136"<?,?> ⬅️ ::Cast(%"val_135") {to=INT64}
    228 |  # node_ReduceSum_228
           %"sum_2"<FLOAT,[s0 + u1,1]> ⬅️ ::ReduceSum(%"mul_121", %"val_136") {noop_with_empty_axes=0, keepdims=False}
    229 |  # node_Identity_229
           %"detach_3"<FLOAT,[s0 + u1,1]> ⬅️ ::Identity(%"sum_2")
    230 |  # node_Identity_230
           %"detach_4"<FLOAT,[s0 + u1,1]> ⬅️ ::Identity(%"detach_3")
    231 |  # node_Identity_231
           %"detach_5"<FLOAT,[s0 + u1,1]> ⬅️ ::Identity(%"detach_4")
    232 |  # node_Cast_232
           %"val_137"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
    233 |  # node_Reshape_233
           %"view_11"<INT64,[s0 + u1,1]> ⬅️ ::Reshape(%"select_8", %"val_137") {allowzero=True}
    234 |  # node_Constant_234
           %"val_138"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    235 |  # node_Reshape_235
           %"val_139"<?,?> ⬅️ ::Reshape(%"add_1226", %"val_138") {allowzero=0}
    236 |  # node_Concat_236
           %"val_140"<?,?> ⬅️ ::Concat(%"val_139", %"val_20") {axis=0}
    237 |  # node_Cast_237
           %"val_141"<?,?> ⬅️ ::Cast(%"val_140") {to=INT64}
    238 |  # node_Abs_238
           %"val_142"<?,?> ⬅️ ::Abs(%"val_141")
    239 |  # node_Expand_239
           %"expand_3"<INT64,[s0 + u1,1]> ⬅️ ::Expand(%"view_11", %"val_142")
    240 |  # node_Constant_240
           %"val_143"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    241 |  # node_Reshape_241
           %"val_144"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_143") {allowzero=0}
    242 |  # node_Concat_242
           %"val_145"<?,?> ⬅️ ::Concat(%"val_144", %"val_20") {axis=0}
    243 |  # node_ConstantOfShape_243
           %"val_146"<?,?> ⬅️ ::ConstantOfShape(%"val_145")
    244 |  # node_CastLike_244
           %"new_zeros_3"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_146", %"detach_5")
    245 |  # node_Shape_245
           %"val_147"<?,?> ⬅️ ::Shape(%"detach_5") {start=0}
    246 |  # node_ConstantOfShape_246
           %"val_148"<?,?> ⬅️ ::ConstantOfShape(%"val_147") {value=Tensor<FLOAT,[1]>(array([-3.4028235e+38], dtype=float32), name=None)}
    247 |  # node_ScatterElements_247
           %"val_149"<?,?> ⬅️ ::ScatterElements(%"new_zeros_3", %"expand_3", %"val_148") {reduction=min, axis=0}
    248 |  # node_ScatterElements_248
           %"scatter_reduce_1"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"val_149", %"expand_3", %"detach_5") {reduction=max, axis=0}
    249 |  # node_Constant_249
           %"val_150"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    250 |  # node_Reshape_250
           %"val_151"<?,?> ⬅️ ::Reshape(%"select_8", %"val_150") {allowzero=0}
    251 |  # node_Cast_251
           %"val_152"<?,?> ⬅️ ::Cast(%"val_151") {to=INT64}
    252 |  # node_Gather_252
           %"index_select_7"<FLOAT,[s0 + u1,1]> ⬅️ ::Gather(%"scatter_reduce_1", %"val_152") {axis=0}
    253 |  # node_Sub_253
           %"sub_99"<FLOAT,[s0 + u1,1]> ⬅️ ::Sub(%"sum_2", %"index_select_7")
    254 |  # node_Exp_254
           %"exp_1"<FLOAT,[s0 + u1,1]> ⬅️ ::Exp(%"sub_99")
    255 |  # node_Cast_255
           %"val_153"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
    256 |  # node_Reshape_256
           %"view_12"<INT64,[s0 + u1,1]> ⬅️ ::Reshape(%"select_8", %"val_153") {allowzero=True}
    257 |  # node_Constant_257
           %"val_154"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    258 |  # node_Reshape_258
           %"val_155"<?,?> ⬅️ ::Reshape(%"add_1226", %"val_154") {allowzero=0}
    259 |  # node_Concat_259
           %"val_156"<?,?> ⬅️ ::Concat(%"val_155", %"val_20") {axis=0}
    260 |  # node_Cast_260
           %"val_157"<?,?> ⬅️ ::Cast(%"val_156") {to=INT64}
    261 |  # node_Abs_261
           %"val_158"<?,?> ⬅️ ::Abs(%"val_157")
    262 |  # node_Expand_262
           %"expand_4"<INT64,[s0 + u1,1]> ⬅️ ::Expand(%"view_12", %"val_158")
    263 |  # node_Constant_263
           %"val_159"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    264 |  # node_Reshape_264
           %"val_160"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_159") {allowzero=0}
    265 |  # node_Concat_265
           %"val_161"<?,?> ⬅️ ::Concat(%"val_160", %"val_20") {axis=0}
    266 |  # node_ConstantOfShape_266
           %"val_162"<?,?> ⬅️ ::ConstantOfShape(%"val_161")
    267 |  # node_CastLike_267
           %"new_zeros_4"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_162", %"exp_1")
    268 |  # node_ScatterElements_268
           %"scatter_add_2"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"new_zeros_4", %"expand_4", %"exp_1") {reduction=add, axis=0}
    269 |  # node_Add_269
           %"add_331"<FLOAT,[s0,1]> ⬅️ ::Add(%"scatter_add_2", %"val_75")
    270 |  # node_Constant_270
           %"val_163"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    271 |  # node_Reshape_271
           %"val_164"<?,?> ⬅️ ::Reshape(%"select_8", %"val_163") {allowzero=0}
    272 |  # node_Cast_272
           %"val_165"<?,?> ⬅️ ::Cast(%"val_164") {to=INT64}
    273 |  # node_Gather_273
           %"index_select_8"<FLOAT,[s0 + u1,1]> ⬅️ ::Gather(%"add_331", %"val_165") {axis=0}
    274 |  # node_Div_274
           %"div_2"<FLOAT,[s0 + u1,1]> ⬅️ ::Div(%"exp_1", %"index_select_8")
    275 |  # node_Identity_275
           %"clone_2"<FLOAT,[s0 + u1,1]> ⬅️ ::Identity(%"div_2")
    276 |  # node_Gather_276
           %"select_10"<INT64,[s0 + u1]> ⬅️ ::Gather(%"cat_1", %"val_5") {axis=0}
    277 |  # node_Gather_277
           %"select_11"<INT64,[s0 + u1]> ⬅️ ::Gather(%"cat_1", %"val_4") {axis=0}
    278 |  # node_Constant_278
           %"val_166"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    279 |  # node_Reshape_279
           %"val_167"<?,?> ⬅️ ::Reshape(%"select_11", %"val_166") {allowzero=0}
    280 |  # node_Cast_280
           %"val_168"<?,?> ⬅️ ::Cast(%"val_167") {to=INT64}
    281 |  # node_Gather_281
           %"index_select_9"<FLOAT,[s0 + u1,1,128]> ⬅️ ::Gather(%"view_8", %"val_168") {axis=0}
    282 |  # node_Unsqueeze_282
           %"unsqueeze_1"<FLOAT,[s0 + u1,1,1]> ⬅️ ::Unsqueeze(%"clone_2", %"val_26")
    283 |  # node_Mul_283
           %"mul_153"<FLOAT,[s0 + u1,1,128]> ⬅️ ::Mul(%"index_select_9", %"unsqueeze_1")
    284 |  # node_Cast_284
           %"val_169"<?,?> ⬅️ ::Cast(%"val_82") {to=INT64}
    285 |  # node_Reshape_285
           %"view_13"<INT64,[s0 + u1,1,1]> ⬅️ ::Reshape(%"select_10", %"val_169") {allowzero=True}
    286 |  # node_Constant_286
           %"val_170"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    287 |  # node_Reshape_287
           %"val_171"<?,?> ⬅️ ::Reshape(%"add_1226", %"val_170") {allowzero=0}
    288 |  # node_Concat_288
           %"val_172"<?,?> ⬅️ ::Concat(%"val_171", %"val_20", %"val_86") {axis=0}
    289 |  # node_Cast_289
           %"val_173"<?,?> ⬅️ ::Cast(%"val_172") {to=INT64}
    290 |  # node_Abs_290
           %"val_174"<?,?> ⬅️ ::Abs(%"val_173")
    291 |  # node_Expand_291
           %"expand_5"<INT64,[s0 + u1,1,128]> ⬅️ ::Expand(%"view_13", %"val_174")
    292 |  # node_Constant_292
           %"val_175"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    293 |  # node_Reshape_293
           %"val_176"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_175") {allowzero=0}
    294 |  # node_Concat_294
           %"val_177"<?,?> ⬅️ ::Concat(%"val_176", %"val_20", %"val_86") {axis=0}
    295 |  # node_ConstantOfShape_295
           %"val_178"<?,?> ⬅️ ::ConstantOfShape(%"val_177")
    296 |  # node_CastLike_296
           %"new_zeros_5"<FLOAT,[s0,1,128]> ⬅️ ::CastLike(%"val_178", %"mul_153")
    297 |  # node_ScatterElements_297
           %"scatter_add_3"<FLOAT,[s0,1,128]> ⬅️ ::ScatterElements(%"new_zeros_5", %"expand_5", %"mul_153") {reduction=add, axis=0}
    298 |  # node_Cast_298
           %"val_179"<?,?> ⬅️ ::Cast(%"val_94") {to=INT64}
    299 |  # node_Reshape_299
           %"view_15"<FLOAT,[s0,128]> ⬅️ ::Reshape(%"scatter_add_3", %"val_179") {allowzero=True}
    300 |  # node_Add_300
           %"add_391"<FLOAT,[s0,128]> ⬅️ ::Add(%"view_15", %"node_processor.graph.convs.1.bias"{...})
    301 |  # node_aten_mean_301
           %"mean_1"<FLOAT,[]> ⬅️ pkg.onnxscript.torch_lib::aten_mean(%"add_391")
    302 |  # node_Sub_302
           %"sub_127"<FLOAT,[s0,128]> ⬅️ ::Sub(%"add_391", %"mean_1")
    303 |  # node_ReduceMean_303
           %"val_180"<?,?> ⬅️ ::ReduceMean(%"sub_127", %"val_96") {noop_with_empty_axes=0, keepdims=True}
    304 |  # node_Sub_304
           %"val_181"<?,?> ⬅️ ::Sub(%"sub_127", %"val_180")
    305 |  # node_Mul_305
           %"val_182"<?,?> ⬅️ ::Mul(%"val_181", %"val_181")
    306 |  # node_ReduceMean_306
           %"var_1"<FLOAT,[]> ⬅️ ::ReduceMean(%"val_182", %"val_96") {noop_with_empty_axes=0, keepdims=False}
    307 |  # node_Sqrt_307
           %"sqrt_1"<FLOAT,[]> ⬅️ ::Sqrt(%"var_1")
    308 |  # node_Add_308
           %"add_398"<FLOAT,[]> ⬅️ ::Add(%"sqrt_1", %"val_100")
    309 |  # node_Div_309
           %"div_3"<FLOAT,[s0,128]> ⬅️ ::Div(%"sub_127", %"add_398")
    310 |  # node_Mul_310
           %"mul_176"<FLOAT,[s0,128]> ⬅️ ::Mul(%"div_3", %"node_processor.graph.norms.1.weight"{...})
    311 |  # node_Add_311
           %"add_405"<FLOAT,[s0,128]> ⬅️ ::Add(%"mul_176", %"node_processor.graph.norms.1.bias"{...})
    312 |  # node_Relu_312
           %"relu_1"<FLOAT,[s0,128]> ⬅️ ::Relu(%"add_405")
    313 |  # node_Identity_313
           %"clone_3"<FLOAT,[s0,128]> ⬅️ ::Identity(%"relu_1")
    314 |  # node_Gemm_314
           %"linear_4"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"clone_3", %"node_processor.graph.convs.2.lin_l.weight"{...}, %"node_processor.graph.convs.2.lin_l.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    315 |  # node_Cast_315
           %"val_183"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
    316 |  # node_Reshape_316
           %"view_16"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear_4", %"val_183") {allowzero=True}
    317 |  # node_Gemm_317
           %"linear_5"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"clone_3", %"node_processor.graph.convs.2.lin_r.weight"{...}, %"node_processor.graph.convs.2.lin_r.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    318 |  # node_Cast_318
           %"val_184"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
    319 |  # node_Reshape_319
           %"view_17"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear_5", %"val_184") {allowzero=True}
    320 |  # node_Gather_320
           %"select_12"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_4") {axis=0}
    321 |  # node_Gather_321
           %"select_13"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_5") {axis=0}
    322 |  # node_Equal_322
           %"val_185"<?,?> ⬅️ ::Equal(%"select_12", %"select_13")
    323 |  # node_Not_323
           %"ne_8"<BOOL,[s1]> ⬅️ ::Not(%"val_185")
    324 |  # node_Cast_324
           %"val_186"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
    325 |  # node_Constant_325
           %"val_187"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    326 |  # node_Reshape_326
           %"val_188"<?,?> ⬅️ ::Reshape(%"val_186", %"val_187") {allowzero=0}
    327 |  # node_Cast_327
           %"val_189"<?,?> ⬅️ ::Cast(%"val_10") {to=INT64}
    328 |  # node_Constant_328
           %"val_190"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    329 |  # node_Reshape_329
           %"val_191"<?,?> ⬅️ ::Reshape(%"val_189", %"val_190") {allowzero=0}
    330 |  # node_Cast_330
           %"val_192"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
    331 |  # node_Constant_331
           %"val_193"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    332 |  # node_Reshape_332
           %"val_194"<?,?> ⬅️ ::Reshape(%"val_192", %"val_193") {allowzero=0}
    333 |  # node_Constant_333
           %"val_195"<?,?> ⬅️ ::Constant() {value_ints=[1]}
    334 |  # node_Slice_334
           %"slice_3"<INT64,[2,s1]> ⬅️ ::Slice(%"edge_index", %"val_188", %"val_191", %"val_194", %"val_195")
    335 |  # node_NonZero_335
           %"val_196"<?,?> ⬅️ ::NonZero(%"ne_8")
    336 |  # node_Transpose_336
           %"val_197"<?,?> ⬅️ ::Transpose(%"val_196") {perm=[1, 0]}
    337 |  # node_Squeeze_337
           %"val_198"<?,?> ⬅️ ::Squeeze(%"val_197", %"val_20")
    338 |  # node_Transpose_338
           %"val_199"<?,?> ⬅️ ::Transpose(%"slice_3") {perm=[1, 0]}
    339 |  # node_Max_339
           %"val_200"<?,?> ⬅️ ::Max(%"val_198")
    340 |  # node_Shape_340
           %"val_201"<?,?> ⬅️ ::Shape(%"val_200") {start=0}
    341 |  # node_Expand_341
           %"val_202"<?,?> ⬅️ ::Expand(%"val_198", %"val_201")
    342 |  # node_Unsqueeze_342
           %"val_203"<?,?> ⬅️ ::Unsqueeze(%"val_202", %"val_26")
    343 |  # node_Concat_343
           %"val_204"<?,?> ⬅️ ::Concat(%"val_203") {axis=-1}
    344 |  # node_GatherND_344
           %"val_205"<?,?> ⬅️ ::GatherND(%"val_199", %"val_204") {batch_dims=0}
    345 |  # node_Transpose_345
           %"index_2"<INT64,[2,u2]> ⬅️ ::Transpose(%"val_205") {perm=[1, 0]}
    346 |  # node_Shape_346
           %"val_206"<?,?> ⬅️ ::Shape(%"index_2") {end=2, start=1}
    347 |  # node_Squeeze_347
           %"sym_size_int_48"<INT64,[]> ⬅️ ::Squeeze(%"val_206")
    348 |  # node_GreaterOrEqual_348
           %"ge_116"<BOOL,[]> ⬅️ ::GreaterOrEqual(%"sym_size_int_48", %"val_4")
    349 |  # node_LessOrEqual_349
           %"le_8"<BOOL,[]> ⬅️ ::LessOrEqual(%"sym_size_int_48", %"val_31")
    350 |  # node_CastLike_350
           %"val_207"<?,?> ⬅️ ::CastLike(%"val_32", %"sym_size_int_44")
    351 |  # node_Range_351
           %"arange_2"<INT64,[s0]> ⬅️ ::Range(%"val_4", %"sym_size_int_44", %"val_207")
    352 |  # node_Cast_352
           %"val_208"<?,?> ⬅️ ::Cast(%"val_34") {to=INT64}
    353 |  # node_Reshape_353
           %"view_18"<INT64,[1,s0]> ⬅️ ::Reshape(%"arange_2", %"val_208") {allowzero=True}
    354 |  # node_Expand_354
           %"val_209"<?,?> ⬅️ ::Expand(%"view_18", %"val_36")
    355 |  # node_Tile_355
           %"repeat_2"<INT64,[2,s0]> ⬅️ ::Tile(%"val_209", %"val_38")
    356 |  # node_Concat_356
           %"cat_2"<INT64,[2,s0 + u2]> ⬅️ ::Concat(%"index_2", %"repeat_2") {axis=1}
    357 |  # node_Gather_357
           %"select_14"<INT64,[s0 + u2]> ⬅️ ::Gather(%"cat_2", %"val_5") {axis=0}
    358 |  # node_Gather_358
           %"select_15"<INT64,[s0 + u2]> ⬅️ ::Gather(%"cat_2", %"val_4") {axis=0}
    359 |  # node_Constant_359
           %"val_210"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    360 |  # node_Reshape_360
           %"val_211"<?,?> ⬅️ ::Reshape(%"select_15", %"val_210") {allowzero=0}
    361 |  # node_Cast_361
           %"val_212"<?,?> ⬅️ ::Cast(%"val_211") {to=INT64}
    362 |  # node_Gather_362
           %"index_select_10"<FLOAT,[s0 + u2,1,128]> ⬅️ ::Gather(%"view_16", %"val_212") {axis=0}
    363 |  # node_Constant_363
           %"val_213"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    364 |  # node_Reshape_364
           %"val_214"<?,?> ⬅️ ::Reshape(%"select_14", %"val_213") {allowzero=0}
    365 |  # node_Cast_365
           %"val_215"<?,?> ⬅️ ::Cast(%"val_214") {to=INT64}
    366 |  # node_Gather_366
           %"index_select_11"<FLOAT,[s0 + u2,1,128]> ⬅️ ::Gather(%"view_17", %"val_215") {axis=0}
    367 |  # node_Add_367
           %"add_1227"<INT64,[]> ⬅️ ::Add(%"sym_size_int_48", %"sym_size_int_44")
    368 |  # node_Add_368
           %"add_463"<FLOAT,[s0 + u2,1,128]> ⬅️ ::Add(%"index_select_11", %"index_select_10")
    369 |  # node_LeakyRelu_369
           %"leaky_relu_2"<FLOAT,[s0 + u2,1,128]> ⬅️ ::LeakyRelu(%"add_463") {alpha=0.2}
    370 |  # node_Mul_370
           %"mul_212"<FLOAT,[s0 + u2,1,128]> ⬅️ ::Mul(%"leaky_relu_2", %"node_processor.graph.convs.2.att"{...})
    371 |  # node_Constant_371
           %"val_216"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    372 |  # node_Reshape_372
           %"val_217"<?,?> ⬅️ ::Reshape(%"val_26", %"val_216") {allowzero=0}
    373 |  # node_Cast_373
           %"val_218"<?,?> ⬅️ ::Cast(%"val_217") {to=INT64}
    374 |  # node_ReduceSum_374
           %"sum_3"<FLOAT,[s0 + u2,1]> ⬅️ ::ReduceSum(%"mul_212", %"val_218") {noop_with_empty_axes=0, keepdims=False}
    375 |  # node_Identity_375
           %"detach_6"<FLOAT,[s0 + u2,1]> ⬅️ ::Identity(%"sum_3")
    376 |  # node_Identity_376
           %"detach_7"<FLOAT,[s0 + u2,1]> ⬅️ ::Identity(%"detach_6")
    377 |  # node_Identity_377
           %"detach_8"<FLOAT,[s0 + u2,1]> ⬅️ ::Identity(%"detach_7")
    378 |  # node_Cast_378
           %"val_219"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
    379 |  # node_Reshape_379
           %"view_19"<INT64,[s0 + u2,1]> ⬅️ ::Reshape(%"select_14", %"val_219") {allowzero=True}
    380 |  # node_Constant_380
           %"val_220"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    381 |  # node_Reshape_381
           %"val_221"<?,?> ⬅️ ::Reshape(%"add_1227", %"val_220") {allowzero=0}
    382 |  # node_Concat_382
           %"val_222"<?,?> ⬅️ ::Concat(%"val_221", %"val_20") {axis=0}
    383 |  # node_Cast_383
           %"val_223"<?,?> ⬅️ ::Cast(%"val_222") {to=INT64}
    384 |  # node_Abs_384
           %"val_224"<?,?> ⬅️ ::Abs(%"val_223")
    385 |  # node_Expand_385
           %"expand_6"<INT64,[s0 + u2,1]> ⬅️ ::Expand(%"view_19", %"val_224")
    386 |  # node_Constant_386
           %"val_225"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    387 |  # node_Reshape_387
           %"val_226"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_225") {allowzero=0}
    388 |  # node_Concat_388
           %"val_227"<?,?> ⬅️ ::Concat(%"val_226", %"val_20") {axis=0}
    389 |  # node_ConstantOfShape_389
           %"val_228"<?,?> ⬅️ ::ConstantOfShape(%"val_227")
    390 |  # node_CastLike_390
           %"new_zeros_6"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_228", %"detach_8")
    391 |  # node_Shape_391
           %"val_229"<?,?> ⬅️ ::Shape(%"detach_8") {start=0}
    392 |  # node_ConstantOfShape_392
           %"val_230"<?,?> ⬅️ ::ConstantOfShape(%"val_229") {value=Tensor<FLOAT,[1]>(array([-3.4028235e+38], dtype=float32), name=None)}
    393 |  # node_ScatterElements_393
           %"val_231"<?,?> ⬅️ ::ScatterElements(%"new_zeros_6", %"expand_6", %"val_230") {reduction=min, axis=0}
    394 |  # node_ScatterElements_394
           %"scatter_reduce_2"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"val_231", %"expand_6", %"detach_8") {reduction=max, axis=0}
    395 |  # node_Constant_395
           %"val_232"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    396 |  # node_Reshape_396
           %"val_233"<?,?> ⬅️ ::Reshape(%"select_14", %"val_232") {allowzero=0}
    397 |  # node_Cast_397
           %"val_234"<?,?> ⬅️ ::Cast(%"val_233") {to=INT64}
    398 |  # node_Gather_398
           %"index_select_12"<FLOAT,[s0 + u2,1]> ⬅️ ::Gather(%"scatter_reduce_2", %"val_234") {axis=0}
    399 |  # node_Sub_399
           %"sub_165"<FLOAT,[s0 + u2,1]> ⬅️ ::Sub(%"sum_3", %"index_select_12")
    400 |  # node_Exp_400
           %"exp_2"<FLOAT,[s0 + u2,1]> ⬅️ ::Exp(%"sub_165")
    401 |  # node_Cast_401
           %"val_235"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
    402 |  # node_Reshape_402
           %"view_20"<INT64,[s0 + u2,1]> ⬅️ ::Reshape(%"select_14", %"val_235") {allowzero=True}
    403 |  # node_Constant_403
           %"val_236"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    404 |  # node_Reshape_404
           %"val_237"<?,?> ⬅️ ::Reshape(%"add_1227", %"val_236") {allowzero=0}
    405 |  # node_Concat_405
           %"val_238"<?,?> ⬅️ ::Concat(%"val_237", %"val_20") {axis=0}
    406 |  # node_Cast_406
           %"val_239"<?,?> ⬅️ ::Cast(%"val_238") {to=INT64}
    407 |  # node_Abs_407
           %"val_240"<?,?> ⬅️ ::Abs(%"val_239")
    408 |  # node_Expand_408
           %"expand_7"<INT64,[s0 + u2,1]> ⬅️ ::Expand(%"view_20", %"val_240")
    409 |  # node_Constant_409
           %"val_241"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    410 |  # node_Reshape_410
           %"val_242"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_241") {allowzero=0}
    411 |  # node_Concat_411
           %"val_243"<?,?> ⬅️ ::Concat(%"val_242", %"val_20") {axis=0}
    412 |  # node_ConstantOfShape_412
           %"val_244"<?,?> ⬅️ ::ConstantOfShape(%"val_243")
    413 |  # node_CastLike_413
           %"new_zeros_7"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_244", %"exp_2")
    414 |  # node_ScatterElements_414
           %"scatter_add_4"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"new_zeros_7", %"expand_7", %"exp_2") {reduction=add, axis=0}
    415 |  # node_Add_415
           %"add_536"<FLOAT,[s0,1]> ⬅️ ::Add(%"scatter_add_4", %"val_75")
    416 |  # node_Constant_416
           %"val_245"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    417 |  # node_Reshape_417
           %"val_246"<?,?> ⬅️ ::Reshape(%"select_14", %"val_245") {allowzero=0}
    418 |  # node_Cast_418
           %"val_247"<?,?> ⬅️ ::Cast(%"val_246") {to=INT64}
    419 |  # node_Gather_419
           %"index_select_13"<FLOAT,[s0 + u2,1]> ⬅️ ::Gather(%"add_536", %"val_247") {axis=0}
    420 |  # node_Div_420
           %"div_4"<FLOAT,[s0 + u2,1]> ⬅️ ::Div(%"exp_2", %"index_select_13")
    421 |  # node_Identity_421
           %"clone_4"<FLOAT,[s0 + u2,1]> ⬅️ ::Identity(%"div_4")
    422 |  # node_Gather_422
           %"select_16"<INT64,[s0 + u2]> ⬅️ ::Gather(%"cat_2", %"val_5") {axis=0}
    423 |  # node_Gather_423
           %"select_17"<INT64,[s0 + u2]> ⬅️ ::Gather(%"cat_2", %"val_4") {axis=0}
    424 |  # node_Constant_424
           %"val_248"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    425 |  # node_Reshape_425
           %"val_249"<?,?> ⬅️ ::Reshape(%"select_17", %"val_248") {allowzero=0}
    426 |  # node_Cast_426
           %"val_250"<?,?> ⬅️ ::Cast(%"val_249") {to=INT64}
    427 |  # node_Gather_427
           %"index_select_14"<FLOAT,[s0 + u2,1,128]> ⬅️ ::Gather(%"view_16", %"val_250") {axis=0}
    428 |  # node_Unsqueeze_428
           %"unsqueeze_2"<FLOAT,[s0 + u2,1,1]> ⬅️ ::Unsqueeze(%"clone_4", %"val_26")
    429 |  # node_Mul_429
           %"mul_244"<FLOAT,[s0 + u2,1,128]> ⬅️ ::Mul(%"index_select_14", %"unsqueeze_2")
    430 |  # node_Cast_430
           %"val_251"<?,?> ⬅️ ::Cast(%"val_82") {to=INT64}
    431 |  # node_Reshape_431
           %"view_21"<INT64,[s0 + u2,1,1]> ⬅️ ::Reshape(%"select_16", %"val_251") {allowzero=True}
    432 |  # node_Constant_432
           %"val_252"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    433 |  # node_Reshape_433
           %"val_253"<?,?> ⬅️ ::Reshape(%"add_1227", %"val_252") {allowzero=0}
    434 |  # node_Concat_434
           %"val_254"<?,?> ⬅️ ::Concat(%"val_253", %"val_20", %"val_86") {axis=0}
    435 |  # node_Cast_435
           %"val_255"<?,?> ⬅️ ::Cast(%"val_254") {to=INT64}
    436 |  # node_Abs_436
           %"val_256"<?,?> ⬅️ ::Abs(%"val_255")
    437 |  # node_Expand_437
           %"expand_8"<INT64,[s0 + u2,1,128]> ⬅️ ::Expand(%"view_21", %"val_256")
    438 |  # node_Constant_438
           %"val_257"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    439 |  # node_Reshape_439
           %"val_258"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_257") {allowzero=0}
    440 |  # node_Concat_440
           %"val_259"<?,?> ⬅️ ::Concat(%"val_258", %"val_20", %"val_86") {axis=0}
    441 |  # node_ConstantOfShape_441
           %"val_260"<?,?> ⬅️ ::ConstantOfShape(%"val_259")
    442 |  # node_CastLike_442
           %"new_zeros_8"<FLOAT,[s0,1,128]> ⬅️ ::CastLike(%"val_260", %"mul_244")
    443 |  # node_ScatterElements_443
           %"scatter_add_5"<FLOAT,[s0,1,128]> ⬅️ ::ScatterElements(%"new_zeros_8", %"expand_8", %"mul_244") {reduction=add, axis=0}
    444 |  # node_Cast_444
           %"val_261"<?,?> ⬅️ ::Cast(%"val_94") {to=INT64}
    445 |  # node_Reshape_445
           %"view_23"<FLOAT,[s0,128]> ⬅️ ::Reshape(%"scatter_add_5", %"val_261") {allowzero=True}
    446 |  # node_Add_446
           %"add_596"<FLOAT,[s0,128]> ⬅️ ::Add(%"view_23", %"node_processor.graph.convs.2.bias"{...})
    447 |  # node_aten_mean_447
           %"mean_2"<FLOAT,[]> ⬅️ pkg.onnxscript.torch_lib::aten_mean(%"add_596")
    448 |  # node_Sub_448
           %"sub_193"<FLOAT,[s0,128]> ⬅️ ::Sub(%"add_596", %"mean_2")
    449 |  # node_ReduceMean_449
           %"val_262"<?,?> ⬅️ ::ReduceMean(%"sub_193", %"val_96") {noop_with_empty_axes=0, keepdims=True}
    450 |  # node_Sub_450
           %"val_263"<?,?> ⬅️ ::Sub(%"sub_193", %"val_262")
    451 |  # node_Mul_451
           %"val_264"<?,?> ⬅️ ::Mul(%"val_263", %"val_263")
    452 |  # node_ReduceMean_452
           %"var_2"<FLOAT,[]> ⬅️ ::ReduceMean(%"val_264", %"val_96") {noop_with_empty_axes=0, keepdims=False}
    453 |  # node_Sqrt_453
           %"sqrt_2"<FLOAT,[]> ⬅️ ::Sqrt(%"var_2")
    454 |  # node_Add_454
           %"add_603"<FLOAT,[]> ⬅️ ::Add(%"sqrt_2", %"val_100")
    455 |  # node_Div_455
           %"div_5"<FLOAT,[s0,128]> ⬅️ ::Div(%"sub_193", %"add_603")
    456 |  # node_Mul_456
           %"mul_267"<FLOAT,[s0,128]> ⬅️ ::Mul(%"div_5", %"node_processor.graph.norms.2.weight"{...})
    457 |  # node_Add_457
           %"add_610"<FLOAT,[s0,128]> ⬅️ ::Add(%"mul_267", %"node_processor.graph.norms.2.bias"{...})
    458 |  # node_Relu_458
           %"relu_2"<FLOAT,[s0,128]> ⬅️ ::Relu(%"add_610")
    459 |  # node_Identity_459
           %"clone_5"<FLOAT,[s0,128]> ⬅️ ::Identity(%"relu_2")
    460 |  # node_Gemm_460
           %"linear_6"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"clone_5", %"node_processor.graph.convs.3.lin_l.weight"{...}, %"node_processor.graph.convs.3.lin_l.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    461 |  # node_Cast_461
           %"val_265"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
    462 |  # node_Reshape_462
           %"view_24"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear_6", %"val_265") {allowzero=True}
    463 |  # node_Gemm_463
           %"linear_7"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"clone_5", %"node_processor.graph.convs.3.lin_r.weight"{...}, %"node_processor.graph.convs.3.lin_r.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    464 |  # node_Cast_464
           %"val_266"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
    465 |  # node_Reshape_465
           %"view_25"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear_7", %"val_266") {allowzero=True}
    466 |  # node_Gather_466
           %"select_18"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_4") {axis=0}
    467 |  # node_Gather_467
           %"select_19"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_5") {axis=0}
    468 |  # node_Equal_468
           %"val_267"<?,?> ⬅️ ::Equal(%"select_18", %"select_19")
    469 |  # node_Not_469
           %"ne_12"<BOOL,[s1]> ⬅️ ::Not(%"val_267")
    470 |  # node_Cast_470
           %"val_268"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
    471 |  # node_Constant_471
           %"val_269"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    472 |  # node_Reshape_472
           %"val_270"<?,?> ⬅️ ::Reshape(%"val_268", %"val_269") {allowzero=0}
    473 |  # node_Cast_473
           %"val_271"<?,?> ⬅️ ::Cast(%"val_10") {to=INT64}
    474 |  # node_Constant_474
           %"val_272"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    475 |  # node_Reshape_475
           %"val_273"<?,?> ⬅️ ::Reshape(%"val_271", %"val_272") {allowzero=0}
    476 |  # node_Cast_476
           %"val_274"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
    477 |  # node_Constant_477
           %"val_275"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    478 |  # node_Reshape_478
           %"val_276"<?,?> ⬅️ ::Reshape(%"val_274", %"val_275") {allowzero=0}
    479 |  # node_Constant_479
           %"val_277"<?,?> ⬅️ ::Constant() {value_ints=[1]}
    480 |  # node_Slice_480
           %"slice_4"<INT64,[2,s1]> ⬅️ ::Slice(%"edge_index", %"val_270", %"val_273", %"val_276", %"val_277")
    481 |  # node_NonZero_481
           %"val_278"<?,?> ⬅️ ::NonZero(%"ne_12")
    482 |  # node_Transpose_482
           %"val_279"<?,?> ⬅️ ::Transpose(%"val_278") {perm=[1, 0]}
    483 |  # node_Squeeze_483
           %"val_280"<?,?> ⬅️ ::Squeeze(%"val_279", %"val_20")
    484 |  # node_Transpose_484
           %"val_281"<?,?> ⬅️ ::Transpose(%"slice_4") {perm=[1, 0]}
    485 |  # node_Max_485
           %"val_282"<?,?> ⬅️ ::Max(%"val_280")
    486 |  # node_Shape_486
           %"val_283"<?,?> ⬅️ ::Shape(%"val_282") {start=0}
    487 |  # node_Expand_487
           %"val_284"<?,?> ⬅️ ::Expand(%"val_280", %"val_283")
    488 |  # node_Unsqueeze_488
           %"val_285"<?,?> ⬅️ ::Unsqueeze(%"val_284", %"val_26")
    489 |  # node_Concat_489
           %"val_286"<?,?> ⬅️ ::Concat(%"val_285") {axis=-1}
    490 |  # node_GatherND_490
           %"val_287"<?,?> ⬅️ ::GatherND(%"val_281", %"val_286") {batch_dims=0}
    491 |  # node_Transpose_491
           %"index_3"<INT64,[2,u3]> ⬅️ ::Transpose(%"val_287") {perm=[1, 0]}
    492 |  # node_Shape_492
           %"val_288"<?,?> ⬅️ ::Shape(%"index_3") {end=2, start=1}
    493 |  # node_Squeeze_493
           %"sym_size_int_49"<INT64,[]> ⬅️ ::Squeeze(%"val_288")
    494 |  # node_GreaterOrEqual_494
           %"ge_117"<BOOL,[]> ⬅️ ::GreaterOrEqual(%"sym_size_int_49", %"val_4")
    495 |  # node_LessOrEqual_495
           %"le_9"<BOOL,[]> ⬅️ ::LessOrEqual(%"sym_size_int_49", %"val_31")
    496 |  # node_CastLike_496
           %"val_289"<?,?> ⬅️ ::CastLike(%"val_32", %"sym_size_int_44")
    497 |  # node_Range_497
           %"arange_3"<INT64,[s0]> ⬅️ ::Range(%"val_4", %"sym_size_int_44", %"val_289")
    498 |  # node_Cast_498
           %"val_290"<?,?> ⬅️ ::Cast(%"val_34") {to=INT64}
    499 |  # node_Reshape_499
           %"view_26"<INT64,[1,s0]> ⬅️ ::Reshape(%"arange_3", %"val_290") {allowzero=True}
    500 |  # node_Expand_500
           %"val_291"<?,?> ⬅️ ::Expand(%"view_26", %"val_36")
    501 |  # node_Tile_501
           %"repeat_3"<INT64,[2,s0]> ⬅️ ::Tile(%"val_291", %"val_38")
    502 |  # node_Concat_502
           %"cat_3"<INT64,[2,s0 + u3]> ⬅️ ::Concat(%"index_3", %"repeat_3") {axis=1}
    503 |  # node_Gather_503
           %"select_20"<INT64,[s0 + u3]> ⬅️ ::Gather(%"cat_3", %"val_5") {axis=0}
    504 |  # node_Gather_504
           %"select_21"<INT64,[s0 + u3]> ⬅️ ::Gather(%"cat_3", %"val_4") {axis=0}
    505 |  # node_Constant_505
           %"val_292"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    506 |  # node_Reshape_506
           %"val_293"<?,?> ⬅️ ::Reshape(%"select_21", %"val_292") {allowzero=0}
    507 |  # node_Cast_507
           %"val_294"<?,?> ⬅️ ::Cast(%"val_293") {to=INT64}
    508 |  # node_Gather_508
           %"index_select_15"<FLOAT,[s0 + u3,1,128]> ⬅️ ::Gather(%"view_24", %"val_294") {axis=0}
    509 |  # node_Constant_509
           %"val_295"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    510 |  # node_Reshape_510
           %"val_296"<?,?> ⬅️ ::Reshape(%"select_20", %"val_295") {allowzero=0}
    511 |  # node_Cast_511
           %"val_297"<?,?> ⬅️ ::Cast(%"val_296") {to=INT64}
    512 |  # node_Gather_512
           %"index_select_16"<FLOAT,[s0 + u3,1,128]> ⬅️ ::Gather(%"view_25", %"val_297") {axis=0}
    513 |  # node_Add_513
           %"add_1228"<INT64,[]> ⬅️ ::Add(%"sym_size_int_49", %"sym_size_int_44")
    514 |  # node_Add_514
           %"add_668"<FLOAT,[s0 + u3,1,128]> ⬅️ ::Add(%"index_select_16", %"index_select_15")
    515 |  # node_LeakyRelu_515
           %"leaky_relu_3"<FLOAT,[s0 + u3,1,128]> ⬅️ ::LeakyRelu(%"add_668") {alpha=0.2}
    516 |  # node_Mul_516
           %"mul_303"<FLOAT,[s0 + u3,1,128]> ⬅️ ::Mul(%"leaky_relu_3", %"node_processor.graph.convs.3.att"{...})
    517 |  # node_Constant_517
           %"val_298"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    518 |  # node_Reshape_518
           %"val_299"<?,?> ⬅️ ::Reshape(%"val_26", %"val_298") {allowzero=0}
    519 |  # node_Cast_519
           %"val_300"<?,?> ⬅️ ::Cast(%"val_299") {to=INT64}
    520 |  # node_ReduceSum_520
           %"sum_4"<FLOAT,[s0 + u3,1]> ⬅️ ::ReduceSum(%"mul_303", %"val_300") {noop_with_empty_axes=0, keepdims=False}
    521 |  # node_Identity_521
           %"detach_9"<FLOAT,[s0 + u3,1]> ⬅️ ::Identity(%"sum_4")
    522 |  # node_Identity_522
           %"detach_10"<FLOAT,[s0 + u3,1]> ⬅️ ::Identity(%"detach_9")
    523 |  # node_Identity_523
           %"detach_11"<FLOAT,[s0 + u3,1]> ⬅️ ::Identity(%"detach_10")
    524 |  # node_Cast_524
           %"val_301"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
    525 |  # node_Reshape_525
           %"view_27"<INT64,[s0 + u3,1]> ⬅️ ::Reshape(%"select_20", %"val_301") {allowzero=True}
    526 |  # node_Constant_526
           %"val_302"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    527 |  # node_Reshape_527
           %"val_303"<?,?> ⬅️ ::Reshape(%"add_1228", %"val_302") {allowzero=0}
    528 |  # node_Concat_528
           %"val_304"<?,?> ⬅️ ::Concat(%"val_303", %"val_20") {axis=0}
    529 |  # node_Cast_529
           %"val_305"<?,?> ⬅️ ::Cast(%"val_304") {to=INT64}
    530 |  # node_Abs_530
           %"val_306"<?,?> ⬅️ ::Abs(%"val_305")
    531 |  # node_Expand_531
           %"expand_9"<INT64,[s0 + u3,1]> ⬅️ ::Expand(%"view_27", %"val_306")
    532 |  # node_Constant_532
           %"val_307"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    533 |  # node_Reshape_533
           %"val_308"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_307") {allowzero=0}
    534 |  # node_Concat_534
           %"val_309"<?,?> ⬅️ ::Concat(%"val_308", %"val_20") {axis=0}
    535 |  # node_ConstantOfShape_535
           %"val_310"<?,?> ⬅️ ::ConstantOfShape(%"val_309")
    536 |  # node_CastLike_536
           %"new_zeros_9"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_310", %"detach_11")
    537 |  # node_Shape_537
           %"val_311"<?,?> ⬅️ ::Shape(%"detach_11") {start=0}
    538 |  # node_ConstantOfShape_538
           %"val_312"<?,?> ⬅️ ::ConstantOfShape(%"val_311") {value=Tensor<FLOAT,[1]>(array([-3.4028235e+38], dtype=float32), name=None)}
    539 |  # node_ScatterElements_539
           %"val_313"<?,?> ⬅️ ::ScatterElements(%"new_zeros_9", %"expand_9", %"val_312") {reduction=min, axis=0}
    540 |  # node_ScatterElements_540
           %"scatter_reduce_3"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"val_313", %"expand_9", %"detach_11") {reduction=max, axis=0}
    541 |  # node_Constant_541
           %"val_314"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    542 |  # node_Reshape_542
           %"val_315"<?,?> ⬅️ ::Reshape(%"select_20", %"val_314") {allowzero=0}
    543 |  # node_Cast_543
           %"val_316"<?,?> ⬅️ ::Cast(%"val_315") {to=INT64}
    544 |  # node_Gather_544
           %"index_select_17"<FLOAT,[s0 + u3,1]> ⬅️ ::Gather(%"scatter_reduce_3", %"val_316") {axis=0}
    545 |  # node_Sub_545
           %"sub_231"<FLOAT,[s0 + u3,1]> ⬅️ ::Sub(%"sum_4", %"index_select_17")
    546 |  # node_Exp_546
           %"exp_3"<FLOAT,[s0 + u3,1]> ⬅️ ::Exp(%"sub_231")
    547 |  # node_Cast_547
           %"val_317"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
    548 |  # node_Reshape_548
           %"view_28"<INT64,[s0 + u3,1]> ⬅️ ::Reshape(%"select_20", %"val_317") {allowzero=True}
    549 |  # node_Constant_549
           %"val_318"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    550 |  # node_Reshape_550
           %"val_319"<?,?> ⬅️ ::Reshape(%"add_1228", %"val_318") {allowzero=0}
    551 |  # node_Concat_551
           %"val_320"<?,?> ⬅️ ::Concat(%"val_319", %"val_20") {axis=0}
    552 |  # node_Cast_552
           %"val_321"<?,?> ⬅️ ::Cast(%"val_320") {to=INT64}
    553 |  # node_Abs_553
           %"val_322"<?,?> ⬅️ ::Abs(%"val_321")
    554 |  # node_Expand_554
           %"expand_10"<INT64,[s0 + u3,1]> ⬅️ ::Expand(%"view_28", %"val_322")
    555 |  # node_Constant_555
           %"val_323"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    556 |  # node_Reshape_556
           %"val_324"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_323") {allowzero=0}
    557 |  # node_Concat_557
           %"val_325"<?,?> ⬅️ ::Concat(%"val_324", %"val_20") {axis=0}
    558 |  # node_ConstantOfShape_558
           %"val_326"<?,?> ⬅️ ::ConstantOfShape(%"val_325")
    559 |  # node_CastLike_559
           %"new_zeros_10"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_326", %"exp_3")
    560 |  # node_ScatterElements_560
           %"scatter_add_6"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"new_zeros_10", %"expand_10", %"exp_3") {reduction=add, axis=0}
    561 |  # node_Add_561
           %"add_741"<FLOAT,[s0,1]> ⬅️ ::Add(%"scatter_add_6", %"val_75")
    562 |  # node_Constant_562
           %"val_327"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    563 |  # node_Reshape_563
           %"val_328"<?,?> ⬅️ ::Reshape(%"select_20", %"val_327") {allowzero=0}
    564 |  # node_Cast_564
           %"val_329"<?,?> ⬅️ ::Cast(%"val_328") {to=INT64}
    565 |  # node_Gather_565
           %"index_select_18"<FLOAT,[s0 + u3,1]> ⬅️ ::Gather(%"add_741", %"val_329") {axis=0}
    566 |  # node_Div_566
           %"div_6"<FLOAT,[s0 + u3,1]> ⬅️ ::Div(%"exp_3", %"index_select_18")
    567 |  # node_Identity_567
           %"clone_6"<FLOAT,[s0 + u3,1]> ⬅️ ::Identity(%"div_6")
    568 |  # node_Gather_568
           %"select_22"<INT64,[s0 + u3]> ⬅️ ::Gather(%"cat_3", %"val_5") {axis=0}
    569 |  # node_Gather_569
           %"select_23"<INT64,[s0 + u3]> ⬅️ ::Gather(%"cat_3", %"val_4") {axis=0}
    570 |  # node_Constant_570
           %"val_330"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    571 |  # node_Reshape_571
           %"val_331"<?,?> ⬅️ ::Reshape(%"select_23", %"val_330") {allowzero=0}
    572 |  # node_Cast_572
           %"val_332"<?,?> ⬅️ ::Cast(%"val_331") {to=INT64}
    573 |  # node_Gather_573
           %"index_select_19"<FLOAT,[s0 + u3,1,128]> ⬅️ ::Gather(%"view_24", %"val_332") {axis=0}
    574 |  # node_Unsqueeze_574
           %"unsqueeze_3"<FLOAT,[s0 + u3,1,1]> ⬅️ ::Unsqueeze(%"clone_6", %"val_26")
    575 |  # node_Mul_575
           %"mul_335"<FLOAT,[s0 + u3,1,128]> ⬅️ ::Mul(%"index_select_19", %"unsqueeze_3")
    576 |  # node_Cast_576
           %"val_333"<?,?> ⬅️ ::Cast(%"val_82") {to=INT64}
    577 |  # node_Reshape_577
           %"view_29"<INT64,[s0 + u3,1,1]> ⬅️ ::Reshape(%"select_22", %"val_333") {allowzero=True}
    578 |  # node_Constant_578
           %"val_334"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    579 |  # node_Reshape_579
           %"val_335"<?,?> ⬅️ ::Reshape(%"add_1228", %"val_334") {allowzero=0}
    580 |  # node_Concat_580
           %"val_336"<?,?> ⬅️ ::Concat(%"val_335", %"val_20", %"val_86") {axis=0}
    581 |  # node_Cast_581
           %"val_337"<?,?> ⬅️ ::Cast(%"val_336") {to=INT64}
    582 |  # node_Abs_582
           %"val_338"<?,?> ⬅️ ::Abs(%"val_337")
    583 |  # node_Expand_583
           %"expand_11"<INT64,[s0 + u3,1,128]> ⬅️ ::Expand(%"view_29", %"val_338")
    584 |  # node_Constant_584
           %"val_339"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    585 |  # node_Reshape_585
           %"val_340"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_339") {allowzero=0}
    586 |  # node_Concat_586
           %"val_341"<?,?> ⬅️ ::Concat(%"val_340", %"val_20", %"val_86") {axis=0}
    587 |  # node_ConstantOfShape_587
           %"val_342"<?,?> ⬅️ ::ConstantOfShape(%"val_341")
    588 |  # node_CastLike_588
           %"new_zeros_11"<FLOAT,[s0,1,128]> ⬅️ ::CastLike(%"val_342", %"mul_335")
    589 |  # node_ScatterElements_589
           %"scatter_add_7"<FLOAT,[s0,1,128]> ⬅️ ::ScatterElements(%"new_zeros_11", %"expand_11", %"mul_335") {reduction=add, axis=0}
    590 |  # node_Cast_590
           %"val_343"<?,?> ⬅️ ::Cast(%"val_94") {to=INT64}
    591 |  # node_Reshape_591
           %"view_31"<FLOAT,[s0,128]> ⬅️ ::Reshape(%"scatter_add_7", %"val_343") {allowzero=True}
    592 |  # node_Add_592
           %"add_801"<FLOAT,[s0,128]> ⬅️ ::Add(%"view_31", %"node_processor.graph.convs.3.bias"{...})
    593 |  # node_aten_mean_593
           %"mean_3"<FLOAT,[]> ⬅️ pkg.onnxscript.torch_lib::aten_mean(%"add_801")
    594 |  # node_Sub_594
           %"sub_259"<FLOAT,[s0,128]> ⬅️ ::Sub(%"add_801", %"mean_3")
    595 |  # node_ReduceMean_595
           %"val_344"<?,?> ⬅️ ::ReduceMean(%"sub_259", %"val_96") {noop_with_empty_axes=0, keepdims=True}
    596 |  # node_Sub_596
           %"val_345"<?,?> ⬅️ ::Sub(%"sub_259", %"val_344")
    597 |  # node_Mul_597
           %"val_346"<?,?> ⬅️ ::Mul(%"val_345", %"val_345")
    598 |  # node_ReduceMean_598
           %"var_3"<FLOAT,[]> ⬅️ ::ReduceMean(%"val_346", %"val_96") {noop_with_empty_axes=0, keepdims=False}
    599 |  # node_Sqrt_599
           %"sqrt_3"<FLOAT,[]> ⬅️ ::Sqrt(%"var_3")
    600 |  # node_Add_600
           %"add_808"<FLOAT,[]> ⬅️ ::Add(%"sqrt_3", %"val_100")
    601 |  # node_Div_601
           %"div_7"<FLOAT,[s0,128]> ⬅️ ::Div(%"sub_259", %"add_808")
    602 |  # node_Mul_602
           %"mul_358"<FLOAT,[s0,128]> ⬅️ ::Mul(%"div_7", %"node_processor.graph.norms.3.weight"{...})
    603 |  # node_Add_603
           %"add_815"<FLOAT,[s0,128]> ⬅️ ::Add(%"mul_358", %"node_processor.graph.norms.3.bias"{...})
    604 |  # node_Relu_604
           %"relu_3"<FLOAT,[s0,128]> ⬅️ ::Relu(%"add_815")
    605 |  # node_Identity_605
           %"clone_7"<FLOAT,[s0,128]> ⬅️ ::Identity(%"relu_3")
    606 |  # node_Gemm_606
           %"linear_8"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"clone_7", %"node_processor.graph.convs.4.lin_l.weight"{...}, %"node_processor.graph.convs.4.lin_l.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    607 |  # node_Cast_607
           %"val_347"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
    608 |  # node_Reshape_608
           %"view_32"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear_8", %"val_347") {allowzero=True}
    609 |  # node_Gemm_609
           %"linear_9"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"clone_7", %"node_processor.graph.convs.4.lin_r.weight"{...}, %"node_processor.graph.convs.4.lin_r.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    610 |  # node_Cast_610
           %"val_348"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
    611 |  # node_Reshape_611
           %"view_33"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear_9", %"val_348") {allowzero=True}
    612 |  # node_Gather_612
           %"select_24"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_4") {axis=0}
    613 |  # node_Gather_613
           %"select_25"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_5") {axis=0}
    614 |  # node_Equal_614
           %"val_349"<?,?> ⬅️ ::Equal(%"select_24", %"select_25")
    615 |  # node_Not_615
           %"ne_16"<BOOL,[s1]> ⬅️ ::Not(%"val_349")
    616 |  # node_Cast_616
           %"val_350"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
    617 |  # node_Constant_617
           %"val_351"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    618 |  # node_Reshape_618
           %"val_352"<?,?> ⬅️ ::Reshape(%"val_350", %"val_351") {allowzero=0}
    619 |  # node_Cast_619
           %"val_353"<?,?> ⬅️ ::Cast(%"val_10") {to=INT64}
    620 |  # node_Constant_620
           %"val_354"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    621 |  # node_Reshape_621
           %"val_355"<?,?> ⬅️ ::Reshape(%"val_353", %"val_354") {allowzero=0}
    622 |  # node_Cast_622
           %"val_356"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
    623 |  # node_Constant_623
           %"val_357"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    624 |  # node_Reshape_624
           %"val_358"<?,?> ⬅️ ::Reshape(%"val_356", %"val_357") {allowzero=0}
    625 |  # node_Constant_625
           %"val_359"<?,?> ⬅️ ::Constant() {value_ints=[1]}
    626 |  # node_Slice_626
           %"slice_5"<INT64,[2,s1]> ⬅️ ::Slice(%"edge_index", %"val_352", %"val_355", %"val_358", %"val_359")
    627 |  # node_NonZero_627
           %"val_360"<?,?> ⬅️ ::NonZero(%"ne_16")
    628 |  # node_Transpose_628
           %"val_361"<?,?> ⬅️ ::Transpose(%"val_360") {perm=[1, 0]}
    629 |  # node_Squeeze_629
           %"val_362"<?,?> ⬅️ ::Squeeze(%"val_361", %"val_20")
    630 |  # node_Transpose_630
           %"val_363"<?,?> ⬅️ ::Transpose(%"slice_5") {perm=[1, 0]}
    631 |  # node_Max_631
           %"val_364"<?,?> ⬅️ ::Max(%"val_362")
    632 |  # node_Shape_632
           %"val_365"<?,?> ⬅️ ::Shape(%"val_364") {start=0}
    633 |  # node_Expand_633
           %"val_366"<?,?> ⬅️ ::Expand(%"val_362", %"val_365")
    634 |  # node_Unsqueeze_634
           %"val_367"<?,?> ⬅️ ::Unsqueeze(%"val_366", %"val_26")
    635 |  # node_Concat_635
           %"val_368"<?,?> ⬅️ ::Concat(%"val_367") {axis=-1}
    636 |  # node_GatherND_636
           %"val_369"<?,?> ⬅️ ::GatherND(%"val_363", %"val_368") {batch_dims=0}
    637 |  # node_Transpose_637
           %"index_4"<INT64,[2,u4]> ⬅️ ::Transpose(%"val_369") {perm=[1, 0]}
    638 |  # node_Shape_638
           %"val_370"<?,?> ⬅️ ::Shape(%"index_4") {end=2, start=1}
    639 |  # node_Squeeze_639
           %"sym_size_int_50"<INT64,[]> ⬅️ ::Squeeze(%"val_370")
    640 |  # node_GreaterOrEqual_640
           %"ge_118"<BOOL,[]> ⬅️ ::GreaterOrEqual(%"sym_size_int_50", %"val_4")
    641 |  # node_LessOrEqual_641
           %"le_10"<BOOL,[]> ⬅️ ::LessOrEqual(%"sym_size_int_50", %"val_31")
    642 |  # node_CastLike_642
           %"val_371"<?,?> ⬅️ ::CastLike(%"val_32", %"sym_size_int_44")
    643 |  # node_Range_643
           %"arange_4"<INT64,[s0]> ⬅️ ::Range(%"val_4", %"sym_size_int_44", %"val_371")
    644 |  # node_Cast_644
           %"val_372"<?,?> ⬅️ ::Cast(%"val_34") {to=INT64}
    645 |  # node_Reshape_645
           %"view_34"<INT64,[1,s0]> ⬅️ ::Reshape(%"arange_4", %"val_372") {allowzero=True}
    646 |  # node_Expand_646
           %"val_373"<?,?> ⬅️ ::Expand(%"view_34", %"val_36")
    647 |  # node_Tile_647
           %"repeat_4"<INT64,[2,s0]> ⬅️ ::Tile(%"val_373", %"val_38")
    648 |  # node_Concat_648
           %"cat_4"<INT64,[2,s0 + u4]> ⬅️ ::Concat(%"index_4", %"repeat_4") {axis=1}
    649 |  # node_Gather_649
           %"select_26"<INT64,[s0 + u4]> ⬅️ ::Gather(%"cat_4", %"val_5") {axis=0}
    650 |  # node_Gather_650
           %"select_27"<INT64,[s0 + u4]> ⬅️ ::Gather(%"cat_4", %"val_4") {axis=0}
    651 |  # node_Constant_651
           %"val_374"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    652 |  # node_Reshape_652
           %"val_375"<?,?> ⬅️ ::Reshape(%"select_27", %"val_374") {allowzero=0}
    653 |  # node_Cast_653
           %"val_376"<?,?> ⬅️ ::Cast(%"val_375") {to=INT64}
    654 |  # node_Gather_654
           %"index_select_20"<FLOAT,[s0 + u4,1,128]> ⬅️ ::Gather(%"view_32", %"val_376") {axis=0}
    655 |  # node_Constant_655
           %"val_377"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    656 |  # node_Reshape_656
           %"val_378"<?,?> ⬅️ ::Reshape(%"select_26", %"val_377") {allowzero=0}
    657 |  # node_Cast_657
           %"val_379"<?,?> ⬅️ ::Cast(%"val_378") {to=INT64}
    658 |  # node_Gather_658
           %"index_select_21"<FLOAT,[s0 + u4,1,128]> ⬅️ ::Gather(%"view_33", %"val_379") {axis=0}
    659 |  # node_Add_659
           %"add_1229"<INT64,[]> ⬅️ ::Add(%"sym_size_int_50", %"sym_size_int_44")
    660 |  # node_Add_660
           %"add_873"<FLOAT,[s0 + u4,1,128]> ⬅️ ::Add(%"index_select_21", %"index_select_20")
    661 |  # node_LeakyRelu_661
           %"leaky_relu_4"<FLOAT,[s0 + u4,1,128]> ⬅️ ::LeakyRelu(%"add_873") {alpha=0.2}
    662 |  # node_Mul_662
           %"mul_394"<FLOAT,[s0 + u4,1,128]> ⬅️ ::Mul(%"leaky_relu_4", %"node_processor.graph.convs.4.att"{...})
    663 |  # node_Constant_663
           %"val_380"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    664 |  # node_Reshape_664
           %"val_381"<?,?> ⬅️ ::Reshape(%"val_26", %"val_380") {allowzero=0}
    665 |  # node_Cast_665
           %"val_382"<?,?> ⬅️ ::Cast(%"val_381") {to=INT64}
    666 |  # node_ReduceSum_666
           %"sum_5"<FLOAT,[s0 + u4,1]> ⬅️ ::ReduceSum(%"mul_394", %"val_382") {noop_with_empty_axes=0, keepdims=False}
    667 |  # node_Identity_667
           %"detach_12"<FLOAT,[s0 + u4,1]> ⬅️ ::Identity(%"sum_5")
    668 |  # node_Identity_668
           %"detach_13"<FLOAT,[s0 + u4,1]> ⬅️ ::Identity(%"detach_12")
    669 |  # node_Identity_669
           %"detach_14"<FLOAT,[s0 + u4,1]> ⬅️ ::Identity(%"detach_13")
    670 |  # node_Cast_670
           %"val_383"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
    671 |  # node_Reshape_671
           %"view_35"<INT64,[s0 + u4,1]> ⬅️ ::Reshape(%"select_26", %"val_383") {allowzero=True}
    672 |  # node_Constant_672
           %"val_384"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    673 |  # node_Reshape_673
           %"val_385"<?,?> ⬅️ ::Reshape(%"add_1229", %"val_384") {allowzero=0}
    674 |  # node_Concat_674
           %"val_386"<?,?> ⬅️ ::Concat(%"val_385", %"val_20") {axis=0}
    675 |  # node_Cast_675
           %"val_387"<?,?> ⬅️ ::Cast(%"val_386") {to=INT64}
    676 |  # node_Abs_676
           %"val_388"<?,?> ⬅️ ::Abs(%"val_387")
    677 |  # node_Expand_677
           %"expand_12"<INT64,[s0 + u4,1]> ⬅️ ::Expand(%"view_35", %"val_388")
    678 |  # node_Constant_678
           %"val_389"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    679 |  # node_Reshape_679
           %"val_390"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_389") {allowzero=0}
    680 |  # node_Concat_680
           %"val_391"<?,?> ⬅️ ::Concat(%"val_390", %"val_20") {axis=0}
    681 |  # node_ConstantOfShape_681
           %"val_392"<?,?> ⬅️ ::ConstantOfShape(%"val_391")
    682 |  # node_CastLike_682
           %"new_zeros_12"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_392", %"detach_14")
    683 |  # node_Shape_683
           %"val_393"<?,?> ⬅️ ::Shape(%"detach_14") {start=0}
    684 |  # node_ConstantOfShape_684
           %"val_394"<?,?> ⬅️ ::ConstantOfShape(%"val_393") {value=Tensor<FLOAT,[1]>(array([-3.4028235e+38], dtype=float32), name=None)}
    685 |  # node_ScatterElements_685
           %"val_395"<?,?> ⬅️ ::ScatterElements(%"new_zeros_12", %"expand_12", %"val_394") {reduction=min, axis=0}
    686 |  # node_ScatterElements_686
           %"scatter_reduce_4"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"val_395", %"expand_12", %"detach_14") {reduction=max, axis=0}
    687 |  # node_Constant_687
           %"val_396"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    688 |  # node_Reshape_688
           %"val_397"<?,?> ⬅️ ::Reshape(%"select_26", %"val_396") {allowzero=0}
    689 |  # node_Cast_689
           %"val_398"<?,?> ⬅️ ::Cast(%"val_397") {to=INT64}
    690 |  # node_Gather_690
           %"index_select_22"<FLOAT,[s0 + u4,1]> ⬅️ ::Gather(%"scatter_reduce_4", %"val_398") {axis=0}
    691 |  # node_Sub_691
           %"sub_297"<FLOAT,[s0 + u4,1]> ⬅️ ::Sub(%"sum_5", %"index_select_22")
    692 |  # node_Exp_692
           %"exp_4"<FLOAT,[s0 + u4,1]> ⬅️ ::Exp(%"sub_297")
    693 |  # node_Cast_693
           %"val_399"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
    694 |  # node_Reshape_694
           %"view_36"<INT64,[s0 + u4,1]> ⬅️ ::Reshape(%"select_26", %"val_399") {allowzero=True}
    695 |  # node_Constant_695
           %"val_400"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    696 |  # node_Reshape_696
           %"val_401"<?,?> ⬅️ ::Reshape(%"add_1229", %"val_400") {allowzero=0}
    697 |  # node_Concat_697
           %"val_402"<?,?> ⬅️ ::Concat(%"val_401", %"val_20") {axis=0}
    698 |  # node_Cast_698
           %"val_403"<?,?> ⬅️ ::Cast(%"val_402") {to=INT64}
    699 |  # node_Abs_699
           %"val_404"<?,?> ⬅️ ::Abs(%"val_403")
    700 |  # node_Expand_700
           %"expand_13"<INT64,[s0 + u4,1]> ⬅️ ::Expand(%"view_36", %"val_404")
    701 |  # node_Constant_701
           %"val_405"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    702 |  # node_Reshape_702
           %"val_406"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_405") {allowzero=0}
    703 |  # node_Concat_703
           %"val_407"<?,?> ⬅️ ::Concat(%"val_406", %"val_20") {axis=0}
    704 |  # node_ConstantOfShape_704
           %"val_408"<?,?> ⬅️ ::ConstantOfShape(%"val_407")
    705 |  # node_CastLike_705
           %"new_zeros_13"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_408", %"exp_4")
    706 |  # node_ScatterElements_706
           %"scatter_add_8"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"new_zeros_13", %"expand_13", %"exp_4") {reduction=add, axis=0}
    707 |  # node_Add_707
           %"add_946"<FLOAT,[s0,1]> ⬅️ ::Add(%"scatter_add_8", %"val_75")
    708 |  # node_Constant_708
           %"val_409"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    709 |  # node_Reshape_709
           %"val_410"<?,?> ⬅️ ::Reshape(%"select_26", %"val_409") {allowzero=0}
    710 |  # node_Cast_710
           %"val_411"<?,?> ⬅️ ::Cast(%"val_410") {to=INT64}
    711 |  # node_Gather_711
           %"index_select_23"<FLOAT,[s0 + u4,1]> ⬅️ ::Gather(%"add_946", %"val_411") {axis=0}
    712 |  # node_Div_712
           %"div_8"<FLOAT,[s0 + u4,1]> ⬅️ ::Div(%"exp_4", %"index_select_23")
    713 |  # node_Identity_713
           %"clone_8"<FLOAT,[s0 + u4,1]> ⬅️ ::Identity(%"div_8")
    714 |  # node_Gather_714
           %"select_28"<INT64,[s0 + u4]> ⬅️ ::Gather(%"cat_4", %"val_5") {axis=0}
    715 |  # node_Gather_715
           %"select_29"<INT64,[s0 + u4]> ⬅️ ::Gather(%"cat_4", %"val_4") {axis=0}
    716 |  # node_Constant_716
           %"val_412"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    717 |  # node_Reshape_717
           %"val_413"<?,?> ⬅️ ::Reshape(%"select_29", %"val_412") {allowzero=0}
    718 |  # node_Cast_718
           %"val_414"<?,?> ⬅️ ::Cast(%"val_413") {to=INT64}
    719 |  # node_Gather_719
           %"index_select_24"<FLOAT,[s0 + u4,1,128]> ⬅️ ::Gather(%"view_32", %"val_414") {axis=0}
    720 |  # node_Unsqueeze_720
           %"unsqueeze_4"<FLOAT,[s0 + u4,1,1]> ⬅️ ::Unsqueeze(%"clone_8", %"val_26")
    721 |  # node_Mul_721
           %"mul_426"<FLOAT,[s0 + u4,1,128]> ⬅️ ::Mul(%"index_select_24", %"unsqueeze_4")
    722 |  # node_Cast_722
           %"val_415"<?,?> ⬅️ ::Cast(%"val_82") {to=INT64}
    723 |  # node_Reshape_723
           %"view_37"<INT64,[s0 + u4,1,1]> ⬅️ ::Reshape(%"select_28", %"val_415") {allowzero=True}
    724 |  # node_Constant_724
           %"val_416"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    725 |  # node_Reshape_725
           %"val_417"<?,?> ⬅️ ::Reshape(%"add_1229", %"val_416") {allowzero=0}
    726 |  # node_Concat_726
           %"val_418"<?,?> ⬅️ ::Concat(%"val_417", %"val_20", %"val_86") {axis=0}
    727 |  # node_Cast_727
           %"val_419"<?,?> ⬅️ ::Cast(%"val_418") {to=INT64}
    728 |  # node_Abs_728
           %"val_420"<?,?> ⬅️ ::Abs(%"val_419")
    729 |  # node_Expand_729
           %"expand_14"<INT64,[s0 + u4,1,128]> ⬅️ ::Expand(%"view_37", %"val_420")
    730 |  # node_Constant_730
           %"val_421"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    731 |  # node_Reshape_731
           %"val_422"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_421") {allowzero=0}
    732 |  # node_Concat_732
           %"val_423"<?,?> ⬅️ ::Concat(%"val_422", %"val_20", %"val_86") {axis=0}
    733 |  # node_ConstantOfShape_733
           %"val_424"<?,?> ⬅️ ::ConstantOfShape(%"val_423")
    734 |  # node_CastLike_734
           %"new_zeros_14"<FLOAT,[s0,1,128]> ⬅️ ::CastLike(%"val_424", %"mul_426")
    735 |  # node_ScatterElements_735
           %"scatter_add_9"<FLOAT,[s0,1,128]> ⬅️ ::ScatterElements(%"new_zeros_14", %"expand_14", %"mul_426") {reduction=add, axis=0}
    736 |  # node_Cast_736
           %"val_425"<?,?> ⬅️ ::Cast(%"val_94") {to=INT64}
    737 |  # node_Reshape_737
           %"view_39"<FLOAT,[s0,128]> ⬅️ ::Reshape(%"scatter_add_9", %"val_425") {allowzero=True}
    738 |  # node_Add_738
           %"add_1006"<FLOAT,[s0,128]> ⬅️ ::Add(%"view_39", %"node_processor.graph.convs.4.bias"{...})
    739 |  # node_aten_mean_739
           %"mean_4"<FLOAT,[]> ⬅️ pkg.onnxscript.torch_lib::aten_mean(%"add_1006")
    740 |  # node_Sub_740
           %"sub_325"<FLOAT,[s0,128]> ⬅️ ::Sub(%"add_1006", %"mean_4")
    741 |  # node_ReduceMean_741
           %"val_426"<?,?> ⬅️ ::ReduceMean(%"sub_325", %"val_96") {noop_with_empty_axes=0, keepdims=True}
    742 |  # node_Sub_742
           %"val_427"<?,?> ⬅️ ::Sub(%"sub_325", %"val_426")
    743 |  # node_Mul_743
           %"val_428"<?,?> ⬅️ ::Mul(%"val_427", %"val_427")
    744 |  # node_ReduceMean_744
           %"var_4"<FLOAT,[]> ⬅️ ::ReduceMean(%"val_428", %"val_96") {noop_with_empty_axes=0, keepdims=False}
    745 |  # node_Sqrt_745
           %"sqrt_4"<FLOAT,[]> ⬅️ ::Sqrt(%"var_4")
    746 |  # node_Add_746
           %"add_1013"<FLOAT,[]> ⬅️ ::Add(%"sqrt_4", %"val_100")
    747 |  # node_Div_747
           %"div_9"<FLOAT,[s0,128]> ⬅️ ::Div(%"sub_325", %"add_1013")
    748 |  # node_Mul_748
           %"mul_449"<FLOAT,[s0,128]> ⬅️ ::Mul(%"div_9", %"node_processor.graph.norms.4.weight"{...})
    749 |  # node_Add_749
           %"add_1020"<FLOAT,[s0,128]> ⬅️ ::Add(%"mul_449", %"node_processor.graph.norms.4.bias"{...})
    750 |  # node_Relu_750
           %"relu_4"<FLOAT,[s0,128]> ⬅️ ::Relu(%"add_1020")
    751 |  # node_Identity_751
           %"clone_9"<FLOAT,[s0,128]> ⬅️ ::Identity(%"relu_4")
    752 |  # node_Gemm_752
           %"linear_10"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"clone_9", %"node_processor.graph.convs.5.lin_l.weight"{...}, %"node_processor.graph.convs.5.lin_l.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    753 |  # node_Cast_753
           %"val_429"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
    754 |  # node_Reshape_754
           %"view_40"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear_10", %"val_429") {allowzero=True}
    755 |  # node_Gemm_755
           %"linear_11"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"clone_9", %"node_processor.graph.convs.5.lin_r.weight"{...}, %"node_processor.graph.convs.5.lin_r.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    756 |  # node_Cast_756
           %"val_430"<?,?> ⬅️ ::Cast(%"val_1") {to=INT64}
    757 |  # node_Reshape_757
           %"view_41"<FLOAT,[s0,1,128]> ⬅️ ::Reshape(%"linear_11", %"val_430") {allowzero=True}
    758 |  # node_Gather_758
           %"select_30"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_4") {axis=0}
    759 |  # node_Gather_759
           %"select_31"<INT64,[s1]> ⬅️ ::Gather(%"edge_index", %"val_5") {axis=0}
    760 |  # node_Equal_760
           %"val_431"<?,?> ⬅️ ::Equal(%"select_30", %"select_31")
    761 |  # node_Not_761
           %"ne_20"<BOOL,[s1]> ⬅️ ::Not(%"val_431")
    762 |  # node_Cast_762
           %"val_432"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
    763 |  # node_Constant_763
           %"val_433"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    764 |  # node_Reshape_764
           %"val_434"<?,?> ⬅️ ::Reshape(%"val_432", %"val_433") {allowzero=0}
    765 |  # node_Cast_765
           %"val_435"<?,?> ⬅️ ::Cast(%"val_10") {to=INT64}
    766 |  # node_Constant_766
           %"val_436"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    767 |  # node_Reshape_767
           %"val_437"<?,?> ⬅️ ::Reshape(%"val_435", %"val_436") {allowzero=0}
    768 |  # node_Cast_768
           %"val_438"<?,?> ⬅️ ::Cast(%"val_4") {to=INT64}
    769 |  # node_Constant_769
           %"val_439"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    770 |  # node_Reshape_770
           %"val_440"<?,?> ⬅️ ::Reshape(%"val_438", %"val_439") {allowzero=0}
    771 |  # node_Constant_771
           %"val_441"<?,?> ⬅️ ::Constant() {value_ints=[1]}
    772 |  # node_Slice_772
           %"slice_6"<INT64,[2,s1]> ⬅️ ::Slice(%"edge_index", %"val_434", %"val_437", %"val_440", %"val_441")
    773 |  # node_NonZero_773
           %"val_442"<?,?> ⬅️ ::NonZero(%"ne_20")
    774 |  # node_Transpose_774
           %"val_443"<?,?> ⬅️ ::Transpose(%"val_442") {perm=[1, 0]}
    775 |  # node_Squeeze_775
           %"val_444"<?,?> ⬅️ ::Squeeze(%"val_443", %"val_20")
    776 |  # node_Transpose_776
           %"val_445"<?,?> ⬅️ ::Transpose(%"slice_6") {perm=[1, 0]}
    777 |  # node_Max_777
           %"val_446"<?,?> ⬅️ ::Max(%"val_444")
    778 |  # node_Shape_778
           %"val_447"<?,?> ⬅️ ::Shape(%"val_446") {start=0}
    779 |  # node_Expand_779
           %"val_448"<?,?> ⬅️ ::Expand(%"val_444", %"val_447")
    780 |  # node_Unsqueeze_780
           %"val_449"<?,?> ⬅️ ::Unsqueeze(%"val_448", %"val_26")
    781 |  # node_Concat_781
           %"val_450"<?,?> ⬅️ ::Concat(%"val_449") {axis=-1}
    782 |  # node_GatherND_782
           %"val_451"<?,?> ⬅️ ::GatherND(%"val_445", %"val_450") {batch_dims=0}
    783 |  # node_Transpose_783
           %"index_5"<INT64,[2,u5]> ⬅️ ::Transpose(%"val_451") {perm=[1, 0]}
    784 |  # node_Shape_784
           %"val_452"<?,?> ⬅️ ::Shape(%"index_5") {end=2, start=1}
    785 |  # node_Squeeze_785
           %"sym_size_int_51"<INT64,[]> ⬅️ ::Squeeze(%"val_452")
    786 |  # node_GreaterOrEqual_786
           %"ge_119"<BOOL,[]> ⬅️ ::GreaterOrEqual(%"sym_size_int_51", %"val_4")
    787 |  # node_LessOrEqual_787
           %"le_11"<BOOL,[]> ⬅️ ::LessOrEqual(%"sym_size_int_51", %"val_31")
    788 |  # node_CastLike_788
           %"val_453"<?,?> ⬅️ ::CastLike(%"val_32", %"sym_size_int_44")
    789 |  # node_Range_789
           %"arange_5"<INT64,[s0]> ⬅️ ::Range(%"val_4", %"sym_size_int_44", %"val_453")
    790 |  # node_Cast_790
           %"val_454"<?,?> ⬅️ ::Cast(%"val_34") {to=INT64}
    791 |  # node_Reshape_791
           %"view_42"<INT64,[1,s0]> ⬅️ ::Reshape(%"arange_5", %"val_454") {allowzero=True}
    792 |  # node_Expand_792
           %"val_455"<?,?> ⬅️ ::Expand(%"view_42", %"val_36")
    793 |  # node_Tile_793
           %"repeat_5"<INT64,[2,s0]> ⬅️ ::Tile(%"val_455", %"val_38")
    794 |  # node_Concat_794
           %"cat_5"<INT64,[2,s0 + u5]> ⬅️ ::Concat(%"index_5", %"repeat_5") {axis=1}
    795 |  # node_Gather_795
           %"select_32"<INT64,[s0 + u5]> ⬅️ ::Gather(%"cat_5", %"val_5") {axis=0}
    796 |  # node_Gather_796
           %"select_33"<INT64,[s0 + u5]> ⬅️ ::Gather(%"cat_5", %"val_4") {axis=0}
    797 |  # node_Constant_797
           %"val_456"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    798 |  # node_Reshape_798
           %"val_457"<?,?> ⬅️ ::Reshape(%"select_33", %"val_456") {allowzero=0}
    799 |  # node_Cast_799
           %"val_458"<?,?> ⬅️ ::Cast(%"val_457") {to=INT64}
    800 |  # node_Gather_800
           %"index_select_25"<FLOAT,[s0 + u5,1,128]> ⬅️ ::Gather(%"view_40", %"val_458") {axis=0}
    801 |  # node_Constant_801
           %"val_459"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    802 |  # node_Reshape_802
           %"val_460"<?,?> ⬅️ ::Reshape(%"select_32", %"val_459") {allowzero=0}
    803 |  # node_Cast_803
           %"val_461"<?,?> ⬅️ ::Cast(%"val_460") {to=INT64}
    804 |  # node_Gather_804
           %"index_select_26"<FLOAT,[s0 + u5,1,128]> ⬅️ ::Gather(%"view_41", %"val_461") {axis=0}
    805 |  # node_Add_805
           %"add_1230"<INT64,[]> ⬅️ ::Add(%"sym_size_int_51", %"sym_size_int_44")
    806 |  # node_Add_806
           %"add_1078"<FLOAT,[s0 + u5,1,128]> ⬅️ ::Add(%"index_select_26", %"index_select_25")
    807 |  # node_LeakyRelu_807
           %"leaky_relu_5"<FLOAT,[s0 + u5,1,128]> ⬅️ ::LeakyRelu(%"add_1078") {alpha=0.2}
    808 |  # node_Mul_808
           %"mul_485"<FLOAT,[s0 + u5,1,128]> ⬅️ ::Mul(%"leaky_relu_5", %"node_processor.graph.convs.5.att"{...})
    809 |  # node_Constant_809
           %"val_462"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    810 |  # node_Reshape_810
           %"val_463"<?,?> ⬅️ ::Reshape(%"val_26", %"val_462") {allowzero=0}
    811 |  # node_Cast_811
           %"val_464"<?,?> ⬅️ ::Cast(%"val_463") {to=INT64}
    812 |  # node_ReduceSum_812
           %"sum_6"<FLOAT,[s0 + u5,1]> ⬅️ ::ReduceSum(%"mul_485", %"val_464") {noop_with_empty_axes=0, keepdims=False}
    813 |  # node_Identity_813
           %"detach_15"<FLOAT,[s0 + u5,1]> ⬅️ ::Identity(%"sum_6")
    814 |  # node_Identity_814
           %"detach_16"<FLOAT,[s0 + u5,1]> ⬅️ ::Identity(%"detach_15")
    815 |  # node_Identity_815
           %"detach_17"<FLOAT,[s0 + u5,1]> ⬅️ ::Identity(%"detach_16")
    816 |  # node_Cast_816
           %"val_465"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
    817 |  # node_Reshape_817
           %"view_43"<INT64,[s0 + u5,1]> ⬅️ ::Reshape(%"select_32", %"val_465") {allowzero=True}
    818 |  # node_Constant_818
           %"val_466"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    819 |  # node_Reshape_819
           %"val_467"<?,?> ⬅️ ::Reshape(%"add_1230", %"val_466") {allowzero=0}
    820 |  # node_Concat_820
           %"val_468"<?,?> ⬅️ ::Concat(%"val_467", %"val_20") {axis=0}
    821 |  # node_Cast_821
           %"val_469"<?,?> ⬅️ ::Cast(%"val_468") {to=INT64}
    822 |  # node_Abs_822
           %"val_470"<?,?> ⬅️ ::Abs(%"val_469")
    823 |  # node_Expand_823
           %"expand_15"<INT64,[s0 + u5,1]> ⬅️ ::Expand(%"view_43", %"val_470")
    824 |  # node_Constant_824
           %"val_471"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    825 |  # node_Reshape_825
           %"val_472"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_471") {allowzero=0}
    826 |  # node_Concat_826
           %"val_473"<?,?> ⬅️ ::Concat(%"val_472", %"val_20") {axis=0}
    827 |  # node_ConstantOfShape_827
           %"val_474"<?,?> ⬅️ ::ConstantOfShape(%"val_473")
    828 |  # node_CastLike_828
           %"new_zeros_15"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_474", %"detach_17")
    829 |  # node_Shape_829
           %"val_475"<?,?> ⬅️ ::Shape(%"detach_17") {start=0}
    830 |  # node_ConstantOfShape_830
           %"val_476"<?,?> ⬅️ ::ConstantOfShape(%"val_475") {value=Tensor<FLOAT,[1]>(array([-3.4028235e+38], dtype=float32), name=None)}
    831 |  # node_ScatterElements_831
           %"val_477"<?,?> ⬅️ ::ScatterElements(%"new_zeros_15", %"expand_15", %"val_476") {reduction=min, axis=0}
    832 |  # node_ScatterElements_832
           %"scatter_reduce_5"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"val_477", %"expand_15", %"detach_17") {reduction=max, axis=0}
    833 |  # node_Constant_833
           %"val_478"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    834 |  # node_Reshape_834
           %"val_479"<?,?> ⬅️ ::Reshape(%"select_32", %"val_478") {allowzero=0}
    835 |  # node_Cast_835
           %"val_480"<?,?> ⬅️ ::Cast(%"val_479") {to=INT64}
    836 |  # node_Gather_836
           %"index_select_27"<FLOAT,[s0 + u5,1]> ⬅️ ::Gather(%"scatter_reduce_5", %"val_480") {axis=0}
    837 |  # node_Sub_837
           %"sub_363"<FLOAT,[s0 + u5,1]> ⬅️ ::Sub(%"sum_6", %"index_select_27")
    838 |  # node_Exp_838
           %"exp_5"<FLOAT,[s0 + u5,1]> ⬅️ ::Exp(%"sub_363")
    839 |  # node_Cast_839
           %"val_481"<?,?> ⬅️ ::Cast(%"val_48") {to=INT64}
    840 |  # node_Reshape_840
           %"view_44"<INT64,[s0 + u5,1]> ⬅️ ::Reshape(%"select_32", %"val_481") {allowzero=True}
    841 |  # node_Constant_841
           %"val_482"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    842 |  # node_Reshape_842
           %"val_483"<?,?> ⬅️ ::Reshape(%"add_1230", %"val_482") {allowzero=0}
    843 |  # node_Concat_843
           %"val_484"<?,?> ⬅️ ::Concat(%"val_483", %"val_20") {axis=0}
    844 |  # node_Cast_844
           %"val_485"<?,?> ⬅️ ::Cast(%"val_484") {to=INT64}
    845 |  # node_Abs_845
           %"val_486"<?,?> ⬅️ ::Abs(%"val_485")
    846 |  # node_Expand_846
           %"expand_16"<INT64,[s0 + u5,1]> ⬅️ ::Expand(%"view_44", %"val_486")
    847 |  # node_Constant_847
           %"val_487"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    848 |  # node_Reshape_848
           %"val_488"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_487") {allowzero=0}
    849 |  # node_Concat_849
           %"val_489"<?,?> ⬅️ ::Concat(%"val_488", %"val_20") {axis=0}
    850 |  # node_ConstantOfShape_850
           %"val_490"<?,?> ⬅️ ::ConstantOfShape(%"val_489")
    851 |  # node_CastLike_851
           %"new_zeros_16"<FLOAT,[s0,1]> ⬅️ ::CastLike(%"val_490", %"exp_5")
    852 |  # node_ScatterElements_852
           %"scatter_add_10"<FLOAT,[s0,1]> ⬅️ ::ScatterElements(%"new_zeros_16", %"expand_16", %"exp_5") {reduction=add, axis=0}
    853 |  # node_Add_853
           %"add_1151"<FLOAT,[s0,1]> ⬅️ ::Add(%"scatter_add_10", %"val_75")
    854 |  # node_Constant_854
           %"val_491"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    855 |  # node_Reshape_855
           %"val_492"<?,?> ⬅️ ::Reshape(%"select_32", %"val_491") {allowzero=0}
    856 |  # node_Cast_856
           %"val_493"<?,?> ⬅️ ::Cast(%"val_492") {to=INT64}
    857 |  # node_Gather_857
           %"index_select_28"<FLOAT,[s0 + u5,1]> ⬅️ ::Gather(%"add_1151", %"val_493") {axis=0}
    858 |  # node_Div_858
           %"div_10"<FLOAT,[s0 + u5,1]> ⬅️ ::Div(%"exp_5", %"index_select_28")
    859 |  # node_Identity_859
           %"clone_10"<FLOAT,[s0 + u5,1]> ⬅️ ::Identity(%"div_10")
    860 |  # node_Gather_860
           %"select_34"<INT64,[s0 + u5]> ⬅️ ::Gather(%"cat_5", %"val_5") {axis=0}
    861 |  # node_Gather_861
           %"select_35"<INT64,[s0 + u5]> ⬅️ ::Gather(%"cat_5", %"val_4") {axis=0}
    862 |  # node_Constant_862
           %"val_494"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    863 |  # node_Reshape_863
           %"val_495"<?,?> ⬅️ ::Reshape(%"select_35", %"val_494") {allowzero=0}
    864 |  # node_Cast_864
           %"val_496"<?,?> ⬅️ ::Cast(%"val_495") {to=INT64}
    865 |  # node_Gather_865
           %"index_select_29"<FLOAT,[s0 + u5,1,128]> ⬅️ ::Gather(%"view_40", %"val_496") {axis=0}
    866 |  # node_Unsqueeze_866
           %"unsqueeze_5"<FLOAT,[s0 + u5,1,1]> ⬅️ ::Unsqueeze(%"clone_10", %"val_26")
    867 |  # node_Mul_867
           %"mul_517"<FLOAT,[s0 + u5,1,128]> ⬅️ ::Mul(%"index_select_29", %"unsqueeze_5")
    868 |  # node_Cast_868
           %"val_497"<?,?> ⬅️ ::Cast(%"val_82") {to=INT64}
    869 |  # node_Reshape_869
           %"view_45"<INT64,[s0 + u5,1,1]> ⬅️ ::Reshape(%"select_34", %"val_497") {allowzero=True}
    870 |  # node_Constant_870
           %"val_498"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    871 |  # node_Reshape_871
           %"val_499"<?,?> ⬅️ ::Reshape(%"add_1230", %"val_498") {allowzero=0}
    872 |  # node_Concat_872
           %"val_500"<?,?> ⬅️ ::Concat(%"val_499", %"val_20", %"val_86") {axis=0}
    873 |  # node_Cast_873
           %"val_501"<?,?> ⬅️ ::Cast(%"val_500") {to=INT64}
    874 |  # node_Abs_874
           %"val_502"<?,?> ⬅️ ::Abs(%"val_501")
    875 |  # node_Expand_875
           %"expand_17"<INT64,[s0 + u5,1,128]> ⬅️ ::Expand(%"view_45", %"val_502")
    876 |  # node_Constant_876
           %"val_503"<?,?> ⬅️ ::Constant() {value=Tensor<INT64,[1]>(array([-1]), name=None)}
    877 |  # node_Reshape_877
           %"val_504"<?,?> ⬅️ ::Reshape(%"sym_size_int_44", %"val_503") {allowzero=0}
    878 |  # node_Concat_878
           %"val_505"<?,?> ⬅️ ::Concat(%"val_504", %"val_20", %"val_86") {axis=0}
    879 |  # node_ConstantOfShape_879
           %"val_506"<?,?> ⬅️ ::ConstantOfShape(%"val_505")
    880 |  # node_CastLike_880
           %"new_zeros_17"<FLOAT,[s0,1,128]> ⬅️ ::CastLike(%"val_506", %"mul_517")
    881 |  # node_ScatterElements_881
           %"scatter_add_11"<FLOAT,[s0,1,128]> ⬅️ ::ScatterElements(%"new_zeros_17", %"expand_17", %"mul_517") {reduction=add, axis=0}
    882 |  # node_Constant_882
           %"val_507"<?,?> ⬅️ ::Constant() {value_ints=[-1]}
    883 |  # node_Reshape_883
           %"val_508"<?,?> ⬅️ ::Reshape(%"val_20", %"val_507") {allowzero=0}
    884 |  # node_ReduceMean_884
           %"mean_5"<FLOAT,[s0,128]> ⬅️ ::ReduceMean(%"scatter_add_11", %"val_508") {noop_with_empty_axes=0, keepdims=False}
    885 |  # node_Add_885
           %"add_1211"<FLOAT,[s0,128]> ⬅️ ::Add(%"mean_5", %"node_processor.graph.convs.5.bias"{...})
    886 |  # node_Add_886
           %"add_1215"<FLOAT,[s0,128]> ⬅️ ::Add(%"embedding", %"add_1211")
    887 |  # node_Gemm_887
           %"linear_12"<FLOAT,[s0,128]> ⬅️ ::Gemm(%"add_1215", %"node_processor.lin.weight"{...}, %"node_processor.lin.bias"{...}) {beta=1.0, transB=True, alpha=1.0, transA=0}
    888 |  # node_Constant_888
           %"val_509"<?,?> ⬅️ ::Constant() {value_ints=[0]}
    889 |  # node__aten_native_batch_norm_inference_onnx_889
           %"xg"<FLOAT,[s0,128]>, %"_native_batch_norm_legit_no_training__1"<FLOAT,[0]>, %"_native_batch_norm_legit_no_training__2"<FLOAT,[0]>, %"val_510"<?,?>, %"val_511"<?,?> ⬅️ pkg.onnxscript.torch_lib::_aten_native_batch_norm_inference_onnx(%"linear_12", %"node_processor.norm.weight"{...}, %"node_processor.norm.bias"{...}, %"node_processor.norm.running_mean"{...}, %"node_processor.norm.running_var"{...}) {momentum=0.9, eps=1e-05}
    return %"xg"<FLOAT,[s0,128]>
}

<
    opset_imports={'': 18},
>
def pkg.onnxscript.torch_lib::aten_mean(
    inputs=(
        %"self"<?,?>
    ),
    outputs=(
        %"return_val"<?,?>
    ),
) {
    0 |  # n0
         %"result"<?,?> ⬅️ ::ReduceMean(%"self")
    1 |  # n1
         %"return_val"<?,?> ⬅️ ::Squeeze(%"result")
    return %"return_val"<?,?>
}

<
    opset_imports={'': 18},
>
def pkg.onnxscript.torch_lib::_aten_native_batch_norm_inference_onnx(
    inputs=(
        %"input"<?,?>,
        %"weight"<?,?>,
        %"bias"<?,?>,
        %"running_mean"<?,?>,
        %"running_var"<?,?>
    ),
    attributes={
        momentum: UNDEFINED,
        eps: UNDEFINED
    }
    outputs=(
        %"norm"<?,?>,
        %"running_mean_fp32"<?,?>,
        %"invstd_1"<?,?>,
        %"return_val3"<?,?>,
        %"return_val4"<?,?>
    ),
) {
     0 |  # n0
          %"norm"<?,?> ⬅️ ::BatchNormalization(%"input", %"weight", %"bias", %"running_mean", %"running_var") {training_mode=0, momentum=@momentum, epsilon=@eps}
     1 |  # n1
          %"const"<?,?> ⬅️ ::Constant() {value=TensorProtoTensor<FLOAT,[]>(array(1., dtype=float32), name='const')}
     2 |  # n2
          %"eps"<?,?> ⬅️ ::Constant() {value_float=@eps}
     3 |  # n3
          %"eps_cast"<?,?> ⬅️ ::CastLike(%"eps", %"running_var")
     4 |  # n4
          %"tmp"<?,?> ⬅️ ::Add(%"running_var", %"eps_cast")
     5 |  # n5
          %"tmp_0"<?,?> ⬅️ ::Sqrt(%"tmp")
     6 |  # n6
          %"const_cast"<?,?> ⬅️ ::CastLike(%"const", %"tmp_0")
     7 |  # n7
          %"invstd"<?,?> ⬅️ ::Div(%"const_cast", %"tmp_0")
     8 |  # n8
          %"running_mean_fp32"<?,?> ⬅️ ::Cast(%"running_mean") {to=1}
     9 |  # n9
          %"invstd_1"<?,?> ⬅️ ::Cast(%"invstd") {to=1}
    10 |  # n10
          %"return_val3"<?,?> ⬅️ ::Identity(%"running_mean")
    11 |  # n11
          %"return_val4"<?,?> ⬅️ ::Identity(%"running_var")
    return %"norm"<?,?>, %"running_mean_fp32"<?,?>, %"invstd_1"<?,?>, %"return_val3"<?,?>, %"return_val4"<?,?>
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

The model has 331392 parameters and 257 buffers (non-trainable parameters).
Number of parameters per dtype:
```python
defaultdict(<class 'int'>, {torch.float32: 331392})
```
Number of buffers per dtype:
```python
defaultdict(<class 'int'>, {torch.float32: 256, torch.int64: 1})
```

Inputs:
- `x`: `TensorMetadata(shape=torch.Size([s0]), dtype=torch.int32, requires_grad=False, stride=(1,), memory_format=torch.contiguous_format, is_quantized=False, qparams={})`
- `edge_index`: `TensorMetadata(shape=torch.Size([2, s1]), dtype=torch.int64, requires_grad=False, stride=(s1, 1), memory_format=torch.contiguous_format, is_quantized=False, qparams={})`

Outputs:
- `getitem`: `TensorMetadata(shape=torch.Size([s0, 128]), dtype=torch.float32, requires_grad=False, stride=(128, 1), memory_format=torch.contiguous_format, is_quantized=False, qparams={})`

The FX graph has 444 nodes in total. Number of FX nodes per op:
- `placeholder`: 62
- `call_function`: 381
- `output`: 1


Of the call_function nodes, the counts of operators used are:

- `aten.view.default`: 41
- `aten.select.int`: 36
- `aten.index_select.default`: 30
- `aten.add.Tensor`: 29
- `aten.detach.default`: 18
- `aten.expand.default`: 18
- `aten.new_zeros.default`: 18
- `aten.mul.Tensor`: 17
- `aten.linear.default`: 13
- `aten.scatter_add.default`: 12
- `aten.sub.Tensor`: 11
- `aten.div.Tensor`: 11
- `aten.clone.default`: 11
- `aten.sym_size.int`: 7
- `aten.ne.Tensor`: 6
- `aten.slice.Tensor`: 6
- `aten.index.Tensor`: 6
- `<built-in function ge>`: 6
- `<built-in function le>`: 6
- `aten.arange.start`: 6
- `aten.repeat.default`: 6
- `aten.cat.default`: 6
- `<built-in function add>`: 6
- `aten.leaky_relu.default`: 6
- `aten.sum.dim_IntList`: 6
- `aten.scatter_reduce.two`: 6
- `aten.exp.default`: 6
- `aten.unsqueeze.default`: 6
- `aten.mean.default`: 5
- `prims.var.default`: 5
- `aten.sqrt.default`: 5
- `aten.relu.default`: 5
- `aten._to_copy.default`: 1
- `aten.embedding.default`: 1
- `aten.mean.dim`: 1
- `aten._native_batch_norm_legit_no_training.default`: 1
- `<built-in function getitem>`: 1

## ONNX Conversion Information

All operators in the model have registered ONNX decompositions.

## Decomposition comparison

Ops exist only in the ExportedProgram before decomposition: `['aten._assert_scalar.default', 'aten.batch_norm.default', 'aten.dropout.default', 'aten.expand_as.default', 'aten.scatter_add_.default', 'aten.scatter_reduce_.two', 'aten.std.default', 'aten.sym_constrain_range_for_size.default', 'aten.to.dtype']`

Ops exist only in the ExportedProgram after decomposition: `['<built-in function add>', '<built-in function getitem>', 'aten._native_batch_norm_legit_no_training.default', 'aten._to_copy.default', 'aten.clone.default', 'aten.expand.default', 'aten.scatter_add.default', 'aten.scatter_reduce.two', 'aten.sqrt.default', 'prims.var.default']`

