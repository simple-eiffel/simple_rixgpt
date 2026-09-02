note
	description: "Char-level GPT (RixGPT) inference in pure Eiffel. Loads model.bin, KV-cached generation."
	author: "ported from a numpy reference validated against PyTorch (logits match 6e-6)"

class
	RIXGPT_MODEL

inherit
	DOUBLE_MATH

create
	make

feature {NONE} -- Initialization

	make (a_path: STRING)
			-- Load model weights + vocab from binary file `a_path`.
		local
			f: RAW_FILE
			mp: MANAGED_POINTER
			i, cp, hdr, rest, nfloats, e, c, ver, dtype, raw: INTEGER
		do
			rng_state := 0x9E3779B9
			create f.make_open_read (a_path)
				-- magic 'RIXG'
			f.read_character; f.read_character; f.read_character; f.read_character
			f.read_integer_32; ver := f.last_integer_32
			if ver >= 2 then f.read_integer_32; dtype := f.last_integer_32 else dtype := 0 end
			f.read_integer_32; n_layer := f.last_integer_32
			f.read_integer_32; n_head := f.last_integer_32
			f.read_integer_32; n_embd := f.last_integer_32
			f.read_integer_32; vocab_size := f.last_integer_32
			f.read_integer_32; block_size := f.last_integer_32
			f.read_integer_32 -- vocab count (== vocab_size)
			head_dim := n_embd // n_head
			create itos.make_filled (0, 0, vocab_size - 1)
			create stoi.make (vocab_size)
			from i := 0 until i >= vocab_size loop
				f.read_integer_32; cp := f.last_integer_32
				itos.put (cp, i)
				stoi.put (i, cp)
				i := i + 1
			end
			hdr := 4 + 4 + 5 * 4 + 4 + vocab_size * 4
			if ver >= 2 then hdr := hdr + 4 end
			rest := f.count - hdr
			create mp.make (rest)
			f.read_to_managed_pointer (mp, 0, rest)
			f.close
				-- widen weights (fp16 or fp32) -> REAL_64 buffer for fast indexed access
			if dtype = 1 then nfloats := rest // 2 else nfloats := rest // 4 end
			create wbuf.make_filled (0.0, 0, nfloats - 1)
			if dtype = 1 then
				from i := 0 until i >= nfloats loop
					raw := mp.read_integer_16 (i * 2).to_integer_32
					if raw < 0 then raw := raw + 65536 end
					wbuf.put (half_to_double (raw), i)
					i := i + 1
				end
			else
				from i := 0 until i >= nfloats loop
					wbuf.put (mp.read_real_32 (i * 4).to_double, i)
					i := i + 1
				end
			end
				-- weight tensor offsets (float indices), in serialized order
			e := n_embd
			off_tok := 0
			off_pos := vocab_size * e
			c := off_pos + block_size * e
			create o_ln1w.make_filled (0, 0, n_layer - 1)
			create o_ln1b.make_filled (0, 0, n_layer - 1)
			create o_inw.make_filled (0, 0, n_layer - 1)
			create o_inb.make_filled (0, 0, n_layer - 1)
			create o_outw.make_filled (0, 0, n_layer - 1)
			create o_outb.make_filled (0, 0, n_layer - 1)
			create o_ln2w.make_filled (0, 0, n_layer - 1)
			create o_ln2b.make_filled (0, 0, n_layer - 1)
			create o_fc1w.make_filled (0, 0, n_layer - 1)
			create o_fc1b.make_filled (0, 0, n_layer - 1)
			create o_fc2w.make_filled (0, 0, n_layer - 1)
			create o_fc2b.make_filled (0, 0, n_layer - 1)
			from i := 0 until i >= n_layer loop
				o_ln1w.put (c, i); c := c + e
				o_ln1b.put (c, i); c := c + e
				o_inw.put (c, i);  c := c + 3 * e * e
				o_inb.put (c, i);  c := c + 3 * e
				o_outw.put (c, i); c := c + e * e
				o_outb.put (c, i); c := c + e
				o_ln2w.put (c, i); c := c + e
				o_ln2b.put (c, i); c := c + e
				o_fc1w.put (c, i); c := c + 4 * e * e
				o_fc1b.put (c, i); c := c + 4 * e
				o_fc2w.put (c, i); c := c + e * 4 * e
				o_fc2b.put (c, i); c := c + e
				i := i + 1
			end
			off_lnfw := c; c := c + e
			off_lnfb := c
				-- KV caches: n_layer * block_size * n_embd doubles each
			create kcache.make_filled (0.0, 0, n_layer * block_size * n_embd - 1)
			create vcache.make_filled (0.0, 0, n_layer * block_size * n_embd - 1)
		end

feature -- Configuration

	n_layer, n_head, n_embd, vocab_size, block_size, head_dim: INTEGER
	itos: ARRAY [INTEGER]
	stoi: HASH_TABLE [INTEGER, INTEGER]

feature -- Generation

	generate (prompt: STRING_32; n_new, top_k: INTEGER; temperature: REAL_64): STRING_32
			-- Continue `prompt` with up to `n_new` sampled characters.
		local
			pos, i, nxt: INTEGER
			logits: ARRAY [REAL_64]
		do
			create Result.make_from_string (prompt)
			create logits.make_filled (0.0, 0, vocab_size - 1)
			pos := 0
			if prompt.is_empty then
				logits := forward (index_of_cp ((' ').code), 0)
				pos := 1
			else
				from i := 1 until i > prompt.count or pos >= block_size loop
					logits := forward (index_of_cp (prompt.item (i).code), pos)
					pos := pos + 1
					i := i + 1
				end
			end
			from i := 0 until i >= n_new or pos >= block_size loop
				nxt := sample (logits, temperature, top_k)
				Result.append_code (itos.item (nxt).to_natural_32)
				logits := forward (nxt, pos)
				pos := pos + 1
				i := i + 1
			end
		end

feature {NONE} -- Forward pass

	forward (tok_id, pos: INTEGER): ARRAY [REAL_64]
			-- Logits over vocabulary after token `tok_id` at position `pos`. Updates KV cache.
		local
			e, l, o, i, kbase: INTEGER
			x, h, q, k, v, ao, h1, h2: ARRAY [REAL_64]
			s: REAL_64
		do
			e := n_embd
			create x.make_filled (0.0, 0, e - 1)
			from i := 0 until i >= e loop
				x.put (wbuf.item (off_tok + tok_id * e + i) + wbuf.item (off_pos + pos * e + i), i)
				i := i + 1
			end
			from l := 0 until l >= n_layer loop
				h := layernorm (x, o_ln1w.item (l), o_ln1b.item (l))
				q := matvec (h, e, e, o_inw.item (l), 0, o_inb.item (l))
				k := matvec (h, e, e, o_inw.item (l), e, o_inb.item (l))
				v := matvec (h, e, e, o_inw.item (l), 2 * e, o_inb.item (l))
				kbase := (l * block_size + pos) * e
				from i := 0 until i >= e loop
					kcache.put (k.item (i), kbase + i)
					vcache.put (v.item (i), kbase + i)
					i := i + 1
				end
				ao := attention (q, l, pos)
				h2 := matvec (ao, e, e, o_outw.item (l), 0, o_outb.item (l))
				from i := 0 until i >= e loop x.put (x.item (i) + h2.item (i), i); i := i + 1 end
				h := layernorm (x, o_ln2w.item (l), o_ln2b.item (l))
				h1 := matvec (h, e, 4 * e, o_fc1w.item (l), 0, o_fc1b.item (l))
				gelu_inplace (h1)
				h2 := matvec (h1, 4 * e, e, o_fc2w.item (l), 0, o_fc2b.item (l))
				from i := 0 until i >= e loop x.put (x.item (i) + h2.item (i), i); i := i + 1 end
				l := l + 1
			end
			x := layernorm (x, off_lnfw, off_lnfb)
			create Result.make_filled (0.0, 0, vocab_size - 1)
			from o := 0 until o >= vocab_size loop
				s := 0.0
				from i := 0 until i >= e loop s := s + x.item (i) * wbuf.item (off_tok + o * e + i); i := i + 1 end
				Result.put (s, o)
				o := o + 1
			end
		end

	attention (q: ARRAY [REAL_64]; layer, pos: INTEGER): ARRAY [REAL_64]
			-- Multi-head causal attention of query `q` against cached K/V for positions 0..pos.
		local
			hidx, s0, p, d, e: INTEGER
			dot, scale: REAL_64
			scores: ARRAY [REAL_64]
		do
			e := n_embd
			create Result.make_filled (0.0, 0, e - 1)
			create scores.make_filled (0.0, 0, pos)
			scale := 1.0 / sqrt (head_dim.to_double)
			from hidx := 0 until hidx >= n_head loop
				s0 := hidx * head_dim
				from p := 0 until p > pos loop
					dot := 0.0
					from d := 0 until d >= head_dim loop
						dot := dot + q.item (s0 + d) * kcache.item ((layer * block_size + p) * e + s0 + d)
						d := d + 1
					end
					scores.put (dot * scale, p)
					p := p + 1
				end
				softmax_range (scores, pos + 1)
				from d := 0 until d >= head_dim loop
					dot := 0.0
					from p := 0 until p > pos loop
						dot := dot + scores.item (p) * vcache.item ((layer * block_size + p) * e + s0 + d)
						p := p + 1
					end
					Result.put (dot, s0 + d)
					d := d + 1
				end
				hidx := hidx + 1
			end
		end

	matvec (inp: ARRAY [REAL_64]; in_dim, out_dim, wbase, row0, bbase: INTEGER): ARRAY [REAL_64]
			-- inp . W[row0 .. row0+out_dim)^T + bias.  W is [total_rows, in_dim] row-major at `wbase`.
		local
			o, i, wrow: INTEGER
			s: REAL_64
		do
			create Result.make_filled (0.0, 0, out_dim - 1)
			from o := 0 until o >= out_dim loop
				s := 0.0
				wrow := wbase + (row0 + o) * in_dim
				from i := 0 until i >= in_dim loop s := s + inp.item (i) * wbuf.item (wrow + i); i := i + 1 end
				if bbase >= 0 then s := s + wbuf.item (bbase + row0 + o) end
				Result.put (s, o)
				o := o + 1
			end
		end

	layernorm (x: ARRAY [REAL_64]; wbase, bbase: INTEGER): ARRAY [REAL_64]
		local
			i, e: INTEGER
			mu, var, inv, d: REAL_64
		do
			e := x.count
			create Result.make_filled (0.0, 0, e - 1)
			mu := 0.0
			from i := 0 until i >= e loop mu := mu + x.item (i); i := i + 1 end
			mu := mu / e
			var := 0.0
			from i := 0 until i >= e loop d := x.item (i) - mu; var := var + d * d; i := i + 1 end
			var := var / e
			inv := 1.0 / sqrt (var + 0.00001)
			from i := 0 until i >= e loop
				Result.put ((x.item (i) - mu) * inv * wbuf.item (wbase + i) + wbuf.item (bbase + i), i)
				i := i + 1
			end
		end

	gelu_inplace (x: ARRAY [REAL_64])
		local
			i: INTEGER
			v: REAL_64
		do
			from i := x.lower until i > x.upper loop
				v := x.item (i)
				x.put (0.5 * v * (1.0 + erf (v * 0.70710678118654752)), i)
				i := i + 1
			end
		end

	softmax_range (a: ARRAY [REAL_64]; n: INTEGER)
		local
			i: INTEGER
			mx, sm: REAL_64
		do
			mx := a.item (0)
			from i := 1 until i >= n loop if a.item (i) > mx then mx := a.item (i) end i := i + 1 end
			sm := 0.0
			from i := 0 until i >= n loop a.put (exp (a.item (i) - mx), i); sm := sm + a.item (i); i := i + 1 end
			from i := 0 until i >= n loop a.put (a.item (i) / sm, i); i := i + 1 end
		end

	erf (x: REAL_64): REAL_64
			-- Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
		local
			t, y, ax, sgn: REAL_64
		do
			if x < 0.0 then sgn := -1.0 else sgn := 1.0 end
			ax := x * sgn
			t := 1.0 / (1.0 + 0.3275911 * ax)
			y := 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp (- ax * ax)
			Result := sgn * y
		end

feature {NONE} -- Sampling

	sample (logits: ARRAY [REAL_64]; temperature: REAL_64; top_k: INTEGER): INTEGER
		local
			i, kk: INTEGER
			l, thresh, sm, r, cum: REAL_64
			work: ARRAY [REAL_64]
		do
			create work.make_filled (0.0, 0, vocab_size - 1)
			from i := 0 until i >= vocab_size loop
				work.put (logits.item (i) / temperature.max (0.000001), i)
				i := i + 1
			end
			kk := top_k
			if kk > 0 and kk < vocab_size then
				thresh := kth_largest (work, kk)
				from i := 0 until i >= vocab_size loop
					if work.item (i) < thresh then work.put (-1.0e30, i) end
					i := i + 1
				end
			end
			l := work.item (0)
			from i := 1 until i >= vocab_size loop if work.item (i) > l then l := work.item (i) end i := i + 1 end
			sm := 0.0
			from i := 0 until i >= vocab_size loop work.put (exp (work.item (i) - l), i); sm := sm + work.item (i); i := i + 1 end
			r := next_uniform * sm
			cum := 0.0
			Result := vocab_size - 1
			from i := 0 until i >= vocab_size loop
				cum := cum + work.item (i)
				if cum >= r then
					Result := i
					i := vocab_size
				else
					i := i + 1
				end
			end
		end

	kth_largest (a: ARRAY [REAL_64]; k: INTEGER): REAL_64
			-- k-th largest value in `a` (copy + partial selection; vocab is small).
		local
			b: ARRAY [REAL_64]
			i, j, mi: INTEGER
			mv, tmp: REAL_64
		do
			b := a.twin
			from j := 0 until j >= k loop
				mi := j; mv := b.item (j)
				from i := j + 1 until i >= b.count loop
					if b.item (i) > mv then mv := b.item (i); mi := i end
					i := i + 1
				end
				tmp := b.item (j); b.put (b.item (mi), j); b.put (tmp, mi)
				j := j + 1
			end
			Result := b.item (k - 1)
		end

	next_uniform: REAL_64
			-- Pseudo-random in [0,1) via xorshift32.
		do
			rng_state := rng_state.bit_xor (rng_state |<< 13)
			rng_state := rng_state.bit_xor (rng_state |>> 17)
			rng_state := rng_state.bit_xor (rng_state |<< 5)
			Result := (rng_state |>> 8).to_integer_32.to_double / 16777216.0
		end

	index_of_cp (cp: INTEGER): INTEGER
		do
			if stoi.has (cp) then Result := stoi.item (cp) else Result := 0 end
		end

feature -- Seeding

	set_seed (s: NATURAL_32)
		do
			if s = 0 then rng_state := 0x9E3779B9 else rng_state := s end
		end

feature {NONE} -- Half-float decoding

	half_to_double (hi: INTEGER): REAL_64
			-- Decode an IEEE-754 binary16 bit pattern `hi` (0..65535) to REAL_64.
		local
			sign, expo, mant: INTEGER
			m: REAL_64
		do
			sign := (hi |>> 15).bit_and (1)
			expo := (hi |>> 10).bit_and (0x1F)
			mant := hi.bit_and (0x3FF)
			if expo = 0 then
				m := (mant.to_double / 1024.0) * pow2 (-14)
			elseif expo = 31 then
				m := 0.0
			else
				m := (1.0 + mant.to_double / 1024.0) * pow2 (expo - 15)
			end
			if sign = 1 then Result := -m else Result := m end
		end

	pow2 (n: INTEGER): REAL_64
			-- Exact 2 ^ n for small integer n.
		local
			i: INTEGER
		do
			Result := 1.0
			if n >= 0 then
				from i := 0 until i >= n loop Result := Result * 2.0; i := i + 1 end
			else
				from i := 0 until i >= -n loop Result := Result / 2.0; i := i + 1 end
			end
		end

feature {NONE} -- Weights & state

	wbuf: ARRAY [REAL_64]
	kcache, vcache: ARRAY [REAL_64]
	off_tok, off_pos, off_lnfw, off_lnfb: INTEGER
	o_ln1w, o_ln1b, o_inw, o_inb, o_outw, o_outb, o_ln2w, o_ln2b, o_fc1w, o_fc1b, o_fc2w, o_fc2b: ARRAY [INTEGER]
	rng_state: NATURAL_32

end
