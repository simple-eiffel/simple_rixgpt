note
	description: "[
		Train RixGPT from scratch in pure Eiffel (M2), on the GPU by default: a port of
		D:\rixgpt\rixgpt.py's train(). Same corpus handling, model, AdamW, warmup-cosine
		LR, gradient clip and evaluation (50 batches per split, dropout off), printing
		rixgpt.py's log line, so the curve can be laid beside its train.log:
			iter   500 | train 1.511 | val 1.201 | lr 9.88e-04 | 20s
		At each evaluation it writes a RIXG checkpoint (f32) that RixGPT.exe runs.

		rixgpt_trainer --data D:\rixgpt\corpus.txt --out D:\rixgpt\eiffel_ckpt.bin
			[--iters 5000] [--eval_every 500] [--eval_batches 50] [--backend gpu|cpu]
	]"
	author: "Larry Rix"

class
	RIXGPT_TRAINER

inherit
	STORAGE_BUDGET_ACCESS

create
	make

feature {NONE} -- Initialization

	make
		local
			cli: SIMPLE_CLI
			f: SIMPLE_TENSOR
			corpus: RIXGPT_CORPUS
			cfg: RIXGPT_CONFIG
		do
			create cli.make
			cli.set_app_info ("rixgpt_trainer", "Train RixGPT in pure Eiffel", "0.1.0")
			cli.add_option_with_default ("d|data", "Corpus (UTF-8 text)", "FILE", "D:\rixgpt\corpus.txt")
			cli.add_option_with_default ("o|out", "RIXG checkpoint to write", "FILE", "D:\rixgpt\eiffel_ckpt.bin")
			cli.add_option_with_default ("i|iters", "Training iterations", "N", "5000")
			cli.add_option_with_default ("e|eval_every", "Evaluate every N iterations", "N", "500")
			cli.add_option_with_default ("b|eval_batches", "Batches per split per evaluation", "N", "50")
			cli.add_option_with_default ("k|backend", "gpu or cpu", "NAME", "gpu")
			cli.add_option_with_default ("t|trace", "Print the training loss every N steps (0 = off)", "N", "0")
			cli.add_option_with_default ("p|pool_mb", "GPU buffer pool limit in MB (0 = no pooling)", "N", "1024")
			cli.add_option_with_default ("m|micro", "Micro-batches per step (gradients merged; same update, less memory)", "N", "2")
			cli.add_option_with_default ("f|profile", "Profile N training steps on the GPU, print the kernel table, then stop (0 = train)", "N", "0")
			cli.parse
			if cli.help_requested then
				cli.print_help
			elseif cli.has_errors then
				cli.print_errors
			else
				if cli.option_value_or_default ("backend", "gpu").same_string ("cpu") then
					create f.make
				else
					create f.make_with_backend (create {VULKAN_BACKEND}.make)
					storage_budget.set_budget (2 * 1024 * 1024 * 1024)
					if attached {VULKAN_BACKEND} f.backend as v then
						v.context.set_pool_limit (cli.integer_option_or_default ("pool_mb", 1024).to_integer_64 * 1024 * 1024)
					end
				end
				create corpus.make (f, cli.option_value_or_default ("data", ""))
				if not corpus.is_loaded then
					print ("[error] " + corpus.last_error.to_string_8 + "%N")
				else
					create cfg.make_recipe (corpus.vocab_size)
					cfg.set_iterations (cli.integer_option_or_default ("iters", 5000))
					cfg.set_eval_every (cli.integer_option_or_default ("eval_every", 500))
					cfg.set_eval_batches (cli.integer_option_or_default ("eval_batches", 50))
					trace_every := cli.integer_option_or_default ("trace", 0)
					micro := cli.integer_option_or_default ("micro", 2).max (1)
					if cli.integer_option_or_default ("profile", 0) > 0 then
						profile (f, corpus, cfg, cli.integer_option_or_default ("profile", 0))
					else
						train (f, corpus, cfg, cli.option_value_or_default ("out", "eiffel_ckpt.bin"))
					end
				end
			end
		end

feature {NONE} -- Settings

	micro: INTEGER
			-- Micro-batches per step: batch_size // micro sequences each.

	trace_every: INTEGER
			-- Print the training loss every this many steps (0: never).

feature {NONE} -- Training

	train (f: SIMPLE_TENSOR; corpus: RIXGPT_CORPUS; cfg: RIXGPT_CONFIG; a_out: STRING)
			-- rixgpt.py's loop: evaluate at every eval_every (and the last) iteration, then step.
		local
			model: RIXGPT_TRAINABLE
			opt: ADAMW_OPTIMIZER
			l_schedule: WARMUP_COSINE_SCHEDULE
			l_train, l_eval_train, l_eval_val: TOKEN_BATCHER
			l_loss: TENSOR
			g: GRAD_STORE
			it, mb: INTEGER
			l_traced: REAL_64
			t0: INTEGER_64
			l_scores: TUPLE [train, val: REAL_64]
			l_writer: RIXG_FILE
		do
			print ("[" + f.backend.name + "] RixGPT in pure Eiffel")
			if attached {VULKAN_BACKEND} f.backend as v then
				print (" on " + v.device_name)
			end
			print ("%N")
			create model.make (f, cfg, create {TENSOR_RANDOM}.make (cfg.seed))
			print ("[model] " + format_fixed (model.parameter_count / 1.0e6, 1) + "M params | vocab=" + cfg.vocab_size.out
				+ " | block=" + cfg.block_size.out + "%N")
			create l_schedule.make (cfg.learning_rate, cfg.warmup, cfg.iterations, 0.1)
			create opt.make (model.parameters, l_schedule, cfg.beta1, cfg.beta2, 1.0e-8, cfg.weight_decay)
			create l_train.make (corpus.train_tokens, cfg.block_size, cfg.batch_size // micro, create {TENSOR_RANDOM}.make (cfg.seed + 1))
			create l_eval_train.make (corpus.train_tokens, cfg.block_size, cfg.batch_size // micro, create {TENSOR_RANDOM}.make (cfg.seed + 2))
			create l_eval_val.make (corpus.val_tokens, cfg.block_size, cfg.batch_size // micro, create {TENSOR_RANDOM}.make (cfg.seed + 3))
			create l_writer.make
			t0 := now_ms
			from it := 0 until it > cfg.iterations loop
				if it \\ cfg.eval_every = 0 or it = cfg.iterations then
					l_scores := [mean_loss (model, l_eval_train, cfg.eval_batches * micro), mean_loss (model, l_eval_val, cfg.eval_batches * micro)]
					print ("iter " + padded (it, 5) + " | train " + format_fixed (l_scores.train, 3) + " | val " + format_fixed (l_scores.val, 3)
						+ " | lr " + scientific (l_schedule.rate (it)) + " | " + ((now_ms - t0) // 1000).out + "s%N")
					l_writer.save (model, corpus.itos, a_out)
					if not l_writer.last_error.is_empty then
						print ("[warning] " + l_writer.last_error.to_string_8 + "%N")
					end
				end
					-- One step = `micro' micro-batches of batch_size // micro: each loss is a mean over
					-- an equal share of the rows, so the mean of their gradients is the gradient of
					-- the whole batch's mean loss (rixgpt.py's update), with a fraction of the memory.
				l_traced := 0.0
				create g.make
				from mb := 1 until mb > micro loop
					l_train.forth
					l_loss := model.loss (l_train.batch.inputs, l_train.batch.targets)
					if trace_every > 0 and then it \\ trace_every = 0 then
						l_traced := l_traced + l_loss.item (0) / micro
					end
					g.merge (l_loss.backward)
					mb := mb + 1
				end
				if micro > 1 then
					g.scale_all ((1.0 / micro).truncated_to_real)
				end
				if trace_every > 0 and then it \\ trace_every = 0 then
					print ("  step " + it.out + " loss " + l_traced.out + " grad norm " + g.global_norm.out
						+ " non-finite " + g.has_non_finite.out + "%N")
				end
				g.clip_global_norm (cfg.clip)
				opt.step (g)
				it := it + 1
			end
			print ("[done] saved " + a_out + " in " + ((now_ms - t0) // 1000).out + "s%N")
		end

	mean_loss (model: RIXGPT_TRAINABLE; b: TOKEN_BATCHER; n: INTEGER): REAL_64
			-- Mean loss over `n' batches with dropout off (rixgpt.py evaluate()).
		local
			k: INTEGER
		do
			model.set_training (False)
			from k := 1 until k > n loop
				b.forth
				Result := Result + model.loss (b.batch.inputs, b.batch.targets).item (0)
				k := k + 1
			end
			Result := Result / n
			model.set_training (True)
		end

feature {NONE} -- Profiling

	profile (f: SIMPLE_TENSOR; corpus: RIXGPT_CORPUS; cfg: RIXGPT_CONFIG; n: INTEGER)
			-- 3 warm-up steps, then `n' steps with GPU kernel timing: wall time per step and
			-- the kernel table (VULKAN_BACKEND.profile_report).
		local
			model: RIXGPT_TRAINABLE
			opt: ADAMW_OPTIMIZER
			l_train: TOKEN_BATCHER
			k: INTEGER
			t0: INTEGER_64
		do
			create model.make (f, cfg, create {TENSOR_RANDOM}.make (cfg.seed))
			create opt.make (model.parameters, create {WARMUP_COSINE_SCHEDULE}.make (cfg.learning_rate, cfg.warmup, cfg.iterations, 0.1),
				cfg.beta1, cfg.beta2, 1.0e-8, cfg.weight_decay)
			create l_train.make (corpus.train_tokens, cfg.block_size, cfg.batch_size // micro, create {TENSOR_RANDOM}.make (cfg.seed + 1))
			from k := 1 until k > 3 loop
				one_step (model, opt, l_train)
				k := k + 1
			end
			if attached {VULKAN_BACKEND} f.backend as v then
				v.reset_profile
				v.set_profiling (True)
			end
			t0 := now_ms
			from k := 1 until k > n loop
				one_step (model, opt, l_train)
				k := k + 1
			end
			print ("[profile] " + n.out + " steps of " + cfg.batch_size.out + " x " + cfg.block_size.out + " (" + micro.out
				+ " micro-batches): " + ((now_ms - t0) // n).out + " ms per step (wall, profiling on)%N")
			if attached {VULKAN_BACKEND} f.backend as v then
				v.set_profiling (False)
				print (v.profile_report)
			end
		end

	one_step (model: RIXGPT_TRAINABLE; opt: ADAMW_OPTIMIZER; b: TOKEN_BATCHER)
			-- One training step of `micro' micro-batches, ending with the loss read back.
		local
			g: GRAD_STORE
			l_loss: TENSOR
			mb: INTEGER
		do
			create g.make
			from mb := 1 until mb > micro loop
				b.forth
				l_loss := model.loss (b.batch.inputs, b.batch.targets)
				g.merge (l_loss.backward)
				last_loss := l_loss.item (0)
				mb := mb + 1
			end
			if micro > 1 then
				g.scale_all ((1.0 / micro).truncated_to_real)
			end
			g.clip_global_norm (1.0)
			opt.step (g)
		end

	last_loss: REAL_32

feature {NONE} -- Formatting

	format_fixed (v: REAL_64; a_decimals: INTEGER): STRING
		local
			fd: FORMAT_DOUBLE
		do
			create fd.make (1, a_decimals)
			Result := fd.formatted (v)
			Result.left_adjust
		end

	scientific (v: REAL_64): STRING
			-- Python's "%.2e" (e.g. 9.88e-04).
		local
			e: INTEGER
			m: REAL_64
		do
			if v = 0.0 then
				Result := "0.00e+00"
			else
				e := {DOUBLE_MATH}.log10 (v.abs).floor
				m := v / (10.0 ^ e)
				if format_fixed (m, 2).same_string ("10.00") then
					e := e + 1
					m := v / (10.0 ^ e)
				end
				Result := format_fixed (m, 2) + "e"
				if e < 0 then Result.append ("-") else Result.append ("+") end
				if e.abs < 10 then Result.append ("0") end
				Result.append (e.abs.out)
			end
		end

	padded (n, w: INTEGER): STRING
		do
			Result := n.out
			from until Result.count >= w loop
				Result.prepend_character (' ')
			end
		end

	now_ms: INTEGER_64
		external
			"C inline use <windows.h>"
		alias
			"LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (EIF_INTEGER_64) ((c.QuadPart * 1000) / f.QuadPart);"
		end

end
