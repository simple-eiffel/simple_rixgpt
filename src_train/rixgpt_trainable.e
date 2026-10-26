note
	description: "[
		RixGPT as a trainable simple_tensor model (rixgpt.py GPT, spec 07: RIXGPT_TRAINABLE).
			x = drop (tok (idx) + pos (0 .. T-1)); x = blocks (x); logits = head (ln_f (x))
		The head is tied to the token embedding. Parameter names follow PyTorch's state_dict
		("tok.weight", "blocks.0.attn.in_proj_weight", "lnf.bias"). Initialization follows
		rixgpt.py's _init: N(0, 0.02) for every Linear and Embedding, zero biases, LayerNorm
		ones/zeros, and nn.MultiheadAttention's xavier-uniform in_proj.
	]"
	author: "Larry Rix"

class
	RIXGPT_TRAINABLE

inherit
	NN_MODULE

create
	make

feature {NONE} -- Initialization

	make (f: SIMPLE_TENSOR; c: RIXGPT_CONFIG; r: TENSOR_RANDOM)
		local
			i: INTEGER
			b: RIXGPT_BLOCK
		do
			make_module
			facade := f
			config := c
			create tok.make_normal (f, c.vocab_size, c.n_embd, 0.02, r)
			create pos.make_normal (f, c.block_size, c.n_embd, 0.02, r)
			create drop.make (c.dropout, r)
			create blocks.make (c.n_layer)
			from i := 1 until i > c.n_layer loop
				create b.make (f, c, r)
				blocks.extend (b)
				i := i + 1
			end
			create lnf.make (f, c.n_embd, 1.0e-5)
			create head.make_tied (f, tok.table, False)
			create positions.make (2)
			register_child ("tok", tok)
			register_child ("pos", pos)
			register_child ("drop", drop)
			across blocks as bl loop
				register_child ("blocks." + (@bl.cursor_index - 1).out, bl)
			end
			register_child ("lnf", lnf)
			register_child ("head", head)
		ensure
			config_set: config = c
		end

feature -- Access

	config: RIXGPT_CONFIG
	tok, pos: NN_EMBEDDING
	drop: NN_DROPOUT
	blocks: ARRAYED_LIST [RIXGPT_BLOCK]
	lnf: NN_LAYER_NORM
	head: NN_LINEAR

	rixg_order: ARRAYED_LIST [TENSOR]
			-- The weights in RIXG file order (the head is tied to tok, so not repeated).
		do
			create Result.make (4 + 12 * blocks.count)
			Result.extend (tok.table)
			Result.extend (pos.table)
			across blocks as b loop
				Result.extend (b.ln1.weight); Result.extend (b.ln1.bias)
				Result.extend (b.attn.in_proj_weight); Result.extend (b.attn.in_proj_bias)
				Result.extend (b.attn.out_proj.weight)
				check attached b.attn.out_proj.bias as ob then Result.extend (ob) end
				Result.extend (b.ln2.weight); Result.extend (b.ln2.bias)
				Result.extend (b.mlp.fc.weight)
				check attached b.mlp.fc.bias as fb then Result.extend (fb) end
				Result.extend (b.mlp.proj.weight)
				check attached b.mlp.proj.bias as pb then Result.extend (pb) end
			end
			Result.extend (lnf.weight)
			Result.extend (lnf.bias)
		end

feature -- Basic operations

	logits (idx: INDEX_TENSOR): TENSOR
			-- (B, T, V) logits for token ids `idx' (B, T).
		require
			batch_by_time: idx.shape.rank = 2
			fits: idx.shape.dim (2) <= config.block_size
			ids_valid: idx.all_in_range (0, config.vocab_size - 1)
		local
			x: TENSOR
		do
			x := drop.forward (tok.forward_indices (idx) + pos.forward_indices (position_ids (idx.shape.dim (2))))
			across blocks as b loop
				x := b.forward (x)
			end
			Result := head.forward (lnf.forward (x))
		ensure
			shaped: Result.shape.rank = 3 and Result.shape.last = config.vocab_size
		end

	loss (idx, targets: INDEX_TENSOR): TENSOR
			-- Mean cross-entropy of next-token predictions.
		require
			batch_by_time: idx.shape.rank = 2
			same_shape: targets.shape ~ idx.shape
			fits: idx.shape.dim (2) <= config.block_size
			ids_valid: idx.all_in_range (0, config.vocab_size - 1)
			targets_valid: targets.all_in_range (0, config.vocab_size - 1)
		do
			Result := logits (idx).cross_entropy (targets, -1)
		ensure
			scalar: Result.is_scalar
		end

feature -- Weights

	load_weights (a_file: RIXG_FILE)
			-- Copy `a_file''s weights into this model (dimensions must agree).
		require
			loaded: a_file.is_loaded
			same_shape: a_file.n_layer = config.n_layer and a_file.n_embd = config.n_embd
				and a_file.vocab_size = config.vocab_size and a_file.block_size = config.block_size
			complete: a_file.weights.count = a_file.expected_weight_count
		local
			l_at, i: INTEGER
			l_values: ARRAY [REAL_32]
		do
			l_at := 1
			across rixg_order as t loop
				create l_values.make_filled (0.0, 1, t.shape.count)
				from i := 1 until i > t.shape.count loop
					l_values [i] := a_file.weights [l_at]
					l_at := l_at + 1
					i := i + 1
				end
				t.backend.upload (l_values, t.storage)
			end
		end

feature {NONE} -- Implementation

	facade: SIMPLE_TENSOR

	positions: HASH_TABLE [INDEX_TENSOR, INTEGER]

	position_ids (t: INTEGER): INDEX_TENSOR
			-- 0 .. t-1.
		local
			l_ids: ARRAY [INTEGER]
			i: INTEGER
		do
			if attached positions.item (t) as p then
				Result := p
			else
				create l_ids.make_filled (0, 1, t)
				from i := 1 until i > t loop
					l_ids [i] := i - 1
					i := i + 1
				end
				Result := facade.indices (l_ids, <<t>>)
				positions.put (Result, t)
			end
		end

invariant
	layers: blocks.count = config.n_layer

end
