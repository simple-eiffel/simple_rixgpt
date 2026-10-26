note
	description: "[
		The training text as rixgpt.py sees it: UTF-8, read in Python's text mode (universal
		newlines: CR LF and a lone CR both become LF), vocabulary = the sorted set of code
		points, tokens = their indices, split 90 / 10 at int (0.9 * len) into train / val.
	]"
	author: "Larry Rix"

class
	RIXGPT_CORPUS

create
	make

feature {NONE} -- Initialization

	make (f: SIMPLE_TENSOR; a_path: READABLE_STRING_GENERAL)
			-- Read and encode the corpus at `a_path'.
		local
			l_file: SIMPLE_FILE
			l_raw, l_text: STRING_32
			l_seen: SPECIAL [BOOLEAN]
			l_codes: ARRAYED_LIST [INTEGER]
			l_ids, l_train, l_val: ARRAY [INTEGER]
			i, c, l_split: INTEGER
		do
			create stoi.make (512)
			create itos.make_empty
			create last_error.make_empty
			create l_file.make (a_path)
			l_raw := l_file.read_text_utf_8
			if l_file.has_error or l_raw.is_empty then
				last_error := {STRING_32} "Cannot read the corpus: " + l_file.last_error
				train_tokens := f.indices (<<0>>, <<1>>)
				val_tokens := train_tokens
			else
				l_text := universal_newlines (l_raw)
				text_length := l_text.count
				create l_seen.make_filled (False, 0x110000)
				from i := 1 until i > l_text.count loop
					l_seen [l_text.code (i).to_integer_32] := True
					i := i + 1
				end
				create l_codes.make (512)
				from c := 0 until c = l_seen.count loop
					if l_seen [c] then
						stoi.put (l_codes.count, c)
						l_codes.extend (c)
					end
					c := c + 1
				end
				itos := l_codes.to_array
				create l_ids.make_filled (0, 1, l_text.count)
				from i := 1 until i > l_text.count loop
					l_ids [i] := stoi.item (l_text.code (i).to_integer_32)
					i := i + 1
				end
				l_split := (0.9 * l_text.count).truncated_to_integer
				l_train := l_ids.subarray (1, l_split)
				l_val := l_ids.subarray (l_split + 1, l_ids.count)
				l_train.rebase (1)
				l_val.rebase (1)
				train_tokens := f.indices (l_train, <<l_train.count>>)
				val_tokens := f.indices (l_val, <<l_val.count>>)
			end
		end

feature -- Access

	itos: ARRAY [INTEGER]
			-- Code point of each token id (1-based array: id k at index k + 1).

	stoi: HASH_TABLE [INTEGER, INTEGER]
			-- Token id of each code point.

	train_tokens, val_tokens: INDEX_TENSOR

	text_length: INTEGER
			-- Characters after newline translation.

	vocab_size: INTEGER
		do
			Result := itos.count
		end

	last_error: STRING_32

feature -- Status report

	is_loaded: BOOLEAN
		do
			Result := last_error.is_empty
		end

feature -- Conversion

	encode (s: READABLE_STRING_32): ARRAY [INTEGER]
			-- Token ids of `s' (unknown characters map to id 0).
		local
			i: INTEGER
		do
			create Result.make_filled (0, 1, s.count)
			from i := 1 until i > s.count loop
				Result [i] := stoi.item (s.code (i).to_integer_32)
				i := i + 1
			end
		end

feature {NONE} -- Implementation

	universal_newlines (s: STRING_32): STRING_32
			-- CR LF -> LF, then lone CR -> LF (Python text mode).
		local
			i: INTEGER
			c: CHARACTER_32
		do
			create Result.make (s.count)
			from i := 1 until i > s.count loop
				c := s [i]
				if c = '%R' then
					Result.append_character ('%N')
					if i < s.count and then s [i + 1] = '%N' then
						i := i + 1
					end
				else
					Result.append_character (c)
				end
				i := i + 1
			end
		end

end
