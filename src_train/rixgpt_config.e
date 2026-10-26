note
	description: "[
		RixGPT's hyperparameters, defaulting to D:\rixgpt\rixgpt.py's recipe (the run that
		reached val 1.201 at iter 500 and 0.782 at 5,000): block 256, batch 64, 6 layers,
		6 heads, n_embd 384, dropout 0.2, lr 1e-3, 5,000 iters, warmup 150, eval every 500
		over 50 batches, AdamW (0.9, 0.99, weight decay 0.1 on every parameter), gradient
		clip 1.0, seed 1337, LR floor 0.1 of peak.
	]"
	author: "Larry Rix"

class
	RIXGPT_CONFIG

create
	make_recipe

feature {NONE} -- Initialization

	make_recipe (a_vocab: INTEGER)
			-- rixgpt.py's defaults for a vocabulary of `a_vocab'.
		require
			positive: a_vocab >= 1
		do
			vocab_size := a_vocab
			block_size := 256
			batch_size := 64
			n_layer := 6
			n_head := 6
			n_embd := 384
			dropout := 0.2
			learning_rate := 1.0e-3
			iterations := 5000
			warmup := 150
			eval_every := 500
			eval_batches := 50
			weight_decay := 0.1
			beta1 := 0.9
			beta2 := 0.99
			clip := 1.0
			seed := 1337
		ensure
			vocab_set: vocab_size = a_vocab
		end

feature -- Access

	vocab_size, block_size, batch_size, n_layer, n_head, n_embd: INTEGER
	dropout, learning_rate, weight_decay, beta1, beta2, clip: REAL_64
	iterations, warmup, eval_every, eval_batches, seed: INTEGER

feature -- Element change

	set_iterations (n: INTEGER)
		require
			positive: n > warmup
		do
			iterations := n
		ensure
			set: iterations = n
		end

	set_eval_every (n: INTEGER)
		require
			positive: n >= 1
		do
			eval_every := n
		ensure
			set: eval_every = n
		end

	set_shape (a_layers, a_heads, a_embd, a_block, a_batch: INTEGER)
			-- A smaller (or larger) model, for tests and experiments.
		require
			positive: a_layers >= 1 and a_heads >= 1 and a_embd >= 1 and a_block >= 1 and a_batch >= 1
			divisible: a_embd \\ a_heads = 0
		do
			n_layer := a_layers; n_head := a_heads; n_embd := a_embd; block_size := a_block; batch_size := a_batch
		ensure
			set: n_layer = a_layers and n_head = a_heads and n_embd = a_embd and block_size = a_block and batch_size = a_batch
		end

	set_eval_batches (n: INTEGER)
		require
			positive: n >= 1
		do
			eval_batches := n
		ensure
			set: eval_batches = n
		end

invariant
	divisible: n_embd \\ n_head = 0
	warmup_before_end: iterations > warmup

end
