note
	description: "[
		The RIXG weights file that RIXGPT_MODEL runs (version 2):
			'RIXG', version 2, dtype (0 = f32, 1 = f16), n_layer, n_head, n_embd, vocab_size,
			block_size, vocab count, vocab code points (int32 each), then the weights in order:
			tok (V x E), pos (block x E), per layer: ln1 w, b; in_proj w (3E x E), b (3E);
			out_proj w (E x E), b; ln2 w, b; fc1 w (4E x E), b (4E); fc2 w (E x 4E), b (E);
			then ln_f w, b. The head is tied to tok. All little-endian.
		`load' reads one; `save' writes one (f32) from a RIXGPT_TRAINABLE.
	]"
	author: "Larry Rix"

class
	RIXG_FILE

create
	make

feature {NONE} -- Initialization

	make
		do
			create last_error.make_empty
			create code_points.make_empty
			create weights.make_empty
		end

feature -- Access

	n_layer, n_head, n_embd, vocab_size, block_size, dtype: INTEGER
	code_points: ARRAY [INTEGER]
			-- Vocabulary (code point of token id k at index k + 1).
	weights: ARRAY [REAL_32]
			-- All weights in file order, widened to f32.
	last_error: STRING_32

	is_loaded: BOOLEAN
		do
			Result := last_error.is_empty and not weights.is_empty
		end

feature -- Reading

	load (a_path: READABLE_STRING_GENERAL)
			-- Read the RIXG file at `a_path'.
		local
			l_file: SIMPLE_FILE
			l_head: MANAGED_POINTER
			l_body: MANAGED_POINTER
			l_size, l_header_bytes: INTEGER_64
			i, l_count, l_version: INTEGER
		do
			create last_error.make_empty
			create l_file.make (a_path)
			l_size := l_file.size
			create l_head.make (36)
			if l_size < 36 or else not l_file.read_block (l_head, 0, 0, 36) then
				last_error := {STRING_32} "Cannot read a RIXG header from " + a_path.to_string_32
			elseif l_head.read_natural_32_le (0) /= 0x47584952 then
				last_error := {STRING_32} "Not a RIXG file (magic): " + a_path.to_string_32
			else
				l_version := l_head.read_integer_32_le (4)
				dtype := l_head.read_integer_32_le (8)
				n_layer := l_head.read_integer_32_le (12)
				n_head := l_head.read_integer_32_le (16)
				n_embd := l_head.read_integer_32_le (20)
				vocab_size := l_head.read_integer_32_le (24)
				block_size := l_head.read_integer_32_le (28)
				if l_version /= 2 or (dtype /= 0 and dtype /= 1) then
					last_error := {STRING_32} "Unsupported RIXG version " + l_version.out + " / dtype " + dtype.out
				else
					l_header_bytes := 36 + 4 * vocab_size.to_integer_64
					create l_body.make ((l_size - 36).to_integer_32)
					if not l_file.read_block (l_body, 0, 36, (l_size - 36).to_integer_32) then
						last_error := {STRING_32} "Cannot read the RIXG body: " + l_file.last_error
					else
						create code_points.make_filled (0, 1, vocab_size)
						from i := 0 until i = vocab_size loop
							code_points [i + 1] := l_body.read_integer_32_le (4 * i)
							i := i + 1
						end
						if dtype = 1 then
							l_count := ((l_size - l_header_bytes) // 2).to_integer_32
						else
							l_count := ((l_size - l_header_bytes) // 4).to_integer_32
						end
						create weights.make_filled (0.0, 1, l_count)
						from i := 0 until i = l_count loop
							if dtype = 1 then
								weights [i + 1] := half_to_real (l_body.read_natural_16_le (4 * vocab_size + 2 * i))
							else
								weights [i + 1] := l_body.read_real_32_le (4 * vocab_size + 4 * i)
							end
							i := i + 1
						end
					end
				end
			end
		end

	expected_weight_count: INTEGER
			-- Floats a well-formed file of these dimensions holds.
		local
			e: INTEGER
		do
			e := n_embd
			Result := vocab_size * e + block_size * e
				+ n_layer * (2 * e + 3 * e * e + 3 * e + e * e + e + 2 * e + 4 * e * e + 4 * e + 4 * e * e + e) + 2 * e
		end

feature -- Writing

	save (a_model: RIXGPT_TRAINABLE; a_code_points: ARRAY [INTEGER]; a_path: READABLE_STRING_GENERAL)
			-- Write `a_model' (f32) with vocabulary `a_code_points'.
		require
			vocab_matches: a_code_points.count = a_model.config.vocab_size
		local
			l_block: MANAGED_POINTER
			l_file: SIMPLE_FILE
			l_pos: INTEGER
			c: RIXGPT_CONFIG
		do
			create last_error.make_empty
			c := a_model.config
			create l_block.make (36 + 4 * c.vocab_size + 4 * a_model.parameter_count.to_integer_32)
			l_block.put_natural_32_le (0x47584952, 0)
			l_block.put_integer_32_le (2, 4)
			l_block.put_integer_32_le (0, 8)
			l_block.put_integer_32_le (c.n_layer, 12)
			l_block.put_integer_32_le (c.n_head, 16)
			l_block.put_integer_32_le (c.n_embd, 20)
			l_block.put_integer_32_le (c.vocab_size, 24)
			l_block.put_integer_32_le (c.block_size, 28)
			l_block.put_integer_32_le (c.vocab_size, 32)
			l_pos := 36
			across a_code_points as cp loop
				l_block.put_integer_32_le (cp, l_pos)
				l_pos := l_pos + 4
			end
			across a_model.rixg_order as t loop
				across t.to_array as v loop
					l_block.put_real_32_le (v, l_pos)
					l_pos := l_pos + 4
				end
			end
			create l_file.make (a_path)
			if not l_file.write_block (l_block, l_pos, False) then
				last_error := {STRING_32} "Cannot write " + a_path.to_string_32 + ": " + l_file.last_error
			end
		end

feature {NONE} -- Implementation

	half_to_real (h: NATURAL_16): REAL_32
			-- IEEE binary16 bits `h' as REAL_32 (Inf/NaN read as 0, as RIXGPT_MODEL does).
		local
			l_sign, l_exp, l_mant: INTEGER
			m: REAL_64
		do
			l_sign := (h |>> 15).to_integer_32
			l_exp := ((h |>> 10) & 0x1F).to_integer_32
			l_mant := (h & 0x3FF).to_integer_32
			if l_exp = 0 then
				m := l_mant / 1024.0 * (2.0 ^ (-14))
			elseif l_exp < 31 then
				m := (1.0 + l_mant / 1024.0) * (2.0 ^ (l_exp - 15))
			end
			if l_sign = 1 then
				m := -m
			end
			Result := m.truncated_to_real
		end

end
