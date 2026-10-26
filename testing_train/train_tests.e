note
	description: "[
		M2 tests: the corpus reproduces rixgpt.py's vocabulary (model.bin's own list);
		the trained weights in model.bin, loaded into RIXGPT_TRAINABLE, give the logits
		RIXGPT_MODEL (validated against PyTorch to 6e-6) gives, on the oracle and on the
		GPU; a tiny model overfits a short text on the GPU.
	]"
	author: "Larry Rix"

class
	TRAIN_TESTS

inherit
	TEST_SET_BASE

feature -- Tests

	test_corpus_matches_recipe
			-- 316 code points, identical to the vocabulary stored in model.bin.
		local
			c: RIXGPT_CORPUS
			r: RIXG_FILE
		do
			create c.make (cpu, Corpus_path)
			assert ("loaded: " + c.last_error.to_string_8, c.is_loaded)
			create r.make
			r.load (Model_path)
			assert ("model.bin read: " + r.last_error.to_string_8, r.is_loaded)
			print ("      corpus: " + c.text_length.out + " characters, vocabulary " + c.vocab_size.out
				+ ", train " + c.train_tokens.count.out + " / val " + c.val_tokens.count.out + "%N")
			assert ("vocabulary size 316", c.vocab_size = 316)
			assert ("same code points as model.bin", across 1 |..| 316 as k all c.itos [k] = r.code_points [k] end)
			assert ("90/10 split", c.train_tokens.count = (0.9 * c.text_length).truncated_to_integer)
		end

	test_rixg_dimensions
			-- model.bin: RIXG v2 f16, 6 x 6 x 384, block 256, and every weight present.
		local
			r: RIXG_FILE
		do
			create r.make
			r.load (Model_path)
			assert ("loaded", r.is_loaded)
			assert ("dims", r.n_layer = 6 and r.n_head = 6 and r.n_embd = 384 and r.vocab_size = 316 and r.block_size = 256 and r.dtype = 1)
			assert ("weight count " + r.weights.count.out + " vs " + r.expected_weight_count.out, r.weights.count = r.expected_weight_count)
		end

	test_parity_oracle
			-- model.bin through RIXGPT_TRAINABLE on the oracle = RIXGPT_MODEL's logits.
		do
			parity (create {SIMPLE_TENSOR}.make_with_backend (create {REFERENCE_BACKEND}), "oracle", 2.0e-3)
		end

	test_parity_gpu
			-- The same on the GPU (float kernels).
		do
			parity (gpu, "GPU", 5.0e-3)
		end

	test_tiny_overfit_gpu
			-- 1 layer, 2 heads, n_embd 32, block 16: 300 AdamW steps on a 60-character text
			-- repeated drive the training loss below 0.3 (it starts near ln 16 = 2.8).
		local
			c: RIXGPT_CONFIG
			r: TENSOR_RANDOM
			m: RIXGPT_TRAINABLE
			opt: ADAMW_OPTIMIZER
			b: TOKEN_BATCHER
			l_text: ARRAY [INTEGER]
			l_loss: TENSOR
			g: GRAD_STORE
			k: INTEGER
			l_first, l_last: REAL_32
		do
			create l_text.make_filled (0, 1, 600)
			from k := 1 until k > 600 loop
				l_text [k] := ((k * 7) \\ 60) \\ 16
				k := k + 1
			end
			create c.make_recipe (16)
			c.set_shape (1, 2, 32, 16, 8)
			create r.make (11)
			create m.make (gpu, c, r)
			create opt.make (m.parameters, create {CONSTANT_SCHEDULE}.make (3.0e-3), 0.9, 0.99, 1.0e-8, 0.0)
			create b.make (gpu.indices (l_text, <<600>>), 16, 8, r)
			from k := 1 until k > 300 loop
				b.forth
				l_loss := m.loss (b.batch.inputs, b.batch.targets)
				if k = 1 then l_first := l_loss.item (0) end
				g := l_loss.backward
				g.clip_global_norm (1.0)
				opt.step (g)
				l_last := l_loss.item (0)
				k := k + 1
			end
			print ("      tiny RixGPT on the GPU: loss " + l_first.out + " -> " + l_last.out + " in 300 steps%N")
			assert ("learned the text: " + l_last.out, l_last < 0.3)
		end

feature {NONE} -- Parity

	parity (f: SIMPLE_TENSOR; a_label: STRING; a_tolerance: REAL_64)
		local
			r: RIXG_FILE
			c: RIXGPT_CONFIG
			m: RIXGPT_TRAINABLE
			ref: RIXGPT_MODEL
			corpus_ids: ARRAY [INTEGER]
			l_logits: TENSOR
			l_ours: ARRAY [REAL_32]
			l_theirs: ARRAY [REAL_64]
			i, t, l_best_ours, l_best_theirs: INTEGER
			l_worst: REAL_64
		do
			create r.make
			r.load (Model_path)
			create c.make_recipe (r.vocab_size)
			create m.make (f, c, create {TENSOR_RANDOM}.make (1))
			m.load_weights (r)
			m.set_training (False)
			corpus_ids := encode (r, Prompt)
			t := corpus_ids.count
			l_logits := m.logits (f.indices (corpus_ids, <<1, t>>))
			l_ours := l_logits.sliced (2, t, 1).reshaped (<<r.vocab_size>>).to_array
			create ref.make (Model_path)
			l_theirs := ref.logits_for (corpus_ids)
			from i := 0 until i = r.vocab_size loop
				l_worst := l_worst.max ((l_ours [i + 1] - l_theirs [i]).abs)
				if l_ours [i + 1] > l_ours [l_best_ours + 1] then l_best_ours := i end
				if l_theirs [i] > l_theirs [l_best_theirs] then l_best_theirs := i end
				i := i + 1
			end
			print ("      parity (" + a_label + "): worst |logit difference| " + l_worst.out + " over " + r.vocab_size.out
				+ " logits after %"" + Prompt + "%"; next character " + l_best_ours.out + " vs " + l_best_theirs.out + "%N")
			assert ("same next character", l_best_ours = l_best_theirs)
			assert ("logits within " + a_tolerance.out + ": " + l_worst.out, l_worst <= a_tolerance)
		end

	encode (r: RIXG_FILE; s: STRING): ARRAY [INTEGER]
			-- Token ids of `s' in `r''s vocabulary.
		local
			i, k: INTEGER
		do
			create Result.make_filled (0, 1, s.count)
			from i := 1 until i > s.count loop
				from k := 1 until k > r.code_points.count or else r.code_points [k] = s [i].code loop
					k := k + 1
				end
				Result [i] := k - 1
				i := i + 1
			end
		end

feature {NONE} -- Fixtures

	Prompt: STRING = "The mind of Christ is"

	Corpus_path: STRING = "D:\rixgpt\corpus.txt"
	Model_path: STRING = "D:\prod\simple_rixgpt\bin\model.bin"

	cpu: SIMPLE_TENSOR
		once
			create Result.make
		end

	gpu: SIMPLE_TENSOR
		once
			create Result.make_with_backend (create {VULKAN_BACKEND}.make)
		end

end
