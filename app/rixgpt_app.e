note
	description: "Interactive CLI for RixGPT: load model.bin, take a prompt, print a continuation."

class
	RIXGPT_APP

inherit
	ARGUMENTS_32
	UTF_CONVERTER

create
	make

feature {NONE} -- Initialization

	make
			-- Entry point: show help, or load weights and run the prompt loop.
		local
			path, arg1, line: STRING
			f: RAW_FILE
			model: RIXGPT_MODEL
			prompt, l_out: STRING_32
			quit: BOOLEAN
			gen_count: INTEGER
			seed: NATURAL_32
			dt: DATE_TIME
		do
			if argument_count >= 1 then arg1 := argument (1).to_string_8 else arg1 := "" end
			if is_help_flag (arg1) then
				print_help
			else
				if arg1.is_empty then path := "model.bin" else path := arg1 end
				create f.make_with_name (path)
				if not f.exists then
					io.put_string ("Cannot find weights file: " + path + "%N")
					io.put_string ("Put model.bin next to this program, or pass its path as the first argument.%N")
					io.put_string ("Run  RixGPT.exe --help  for usage.%N")
				else
					io.put_string ("%N  RixGPT - a tiny language model trained on Larry Rix's frameworks.%N")
					io.put_string ("  It writes in his VOICE; it does not KNOW facts. Just for fun.%N")
					io.put_string ("  Type a few words and press Enter. Type  quit  to exit.  (--help for more)%N")
					io.put_string ("  Loading model (a few seconds)...%N")
					io.output.flush
					create model.make (path)
					create dt.make_now
					from until quit loop
						io.put_string ("%Nyou> ")
						io.read_line
						line := io.last_string.twin
						line.right_adjust
						line.left_adjust
						if line.is_empty or else line.as_lower.same_string ("quit") or else line.as_lower.same_string ("exit") then
							quit := True
						else
							gen_count := gen_count + 1
							seed := (dt.time.hour * 3600 + dt.time.minute * 60 + dt.time.second).to_natural_32
							seed := seed.bit_xor (line.hash_code.to_natural_32)
							seed := seed.bit_xor (gen_count.to_natural_32 |<< 20)
							model.set_seed (seed)
							prompt := utf_8_string_8_to_string_32 (line)
							l_out := model.generate (prompt, 400, 200, 0.85)
							io.put_string ("%N")
							io.put_string (utf_32_string_to_utf_8_string_8 (l_out))
							io.put_string ("%N")
						end
					end
					io.put_string ("%Nbye.%N")
				end
			end
		end

feature {NONE} -- Help

	is_help_flag (a: STRING): BOOLEAN
			-- Does `a` request the help screen?
		local
			s: STRING
		do
			s := a.as_lower
			Result := s.same_string ("--help") or else s.same_string ("-h") or else s.same_string ("help")
				or else s.same_string ("/?") or else s.same_string ("-?") or else s.same_string ("/help")
		end

	print_help
			-- Print usage instructions.
		do
			io.put_string ("%N")
			io.put_string ("RixGPT - a tiny character-level language model that writes in Larry Rix's voice.%N")
			io.put_string ("It runs entirely offline on your PC. It imitates his STYLE, not facts - everything%N")
			io.put_string ("it writes is invented. A novelty, not a source of truth.%N")
			io.put_string ("%N")
			io.put_string ("USAGE%N")
			io.put_string ("  RixGPT.exe [MODEL_FILE]%N")
			io.put_string ("  RixGPT.exe --help%N")
			io.put_string ("%N")
			io.put_string ("  MODEL_FILE   Optional path to the weights file. Defaults to  model.bin  in the%N")
			io.put_string ("               current folder. Keep RixGPT.exe and model.bin together.%N")
			io.put_string ("%N")
			io.put_string ("HOW TO USE%N")
			io.put_string ("  1. Open a Command Prompt and go to the folder holding RixGPT.exe and model.bin:%N")
			io.put_string ("        cd /d D:\path\to\the\folder%N")
			io.put_string ("  2. Start it:%N")
			io.put_string ("        RixGPT.exe%N")
			io.put_string ("  3. At the  you>  prompt, type a few words (for example:  The mind of Christ is )%N")
			io.put_string ("     and press Enter. It continues your text in Larry's style.%N")
			io.put_string ("  4. Enter another prompt, or type  quit  (or  exit ) to stop.%N")
			io.put_string ("%N")
			io.put_string ("  Tip: double-click  'Start RixGPT.bat'  to launch with correct special characters%N")
			io.put_string ("       (em-dashes, Hebrew) already enabled.%N")
			io.put_string ("%N")
			io.put_string ("NOTES%N")
			io.put_string ("  * No internet, no accounts, nothing installed. Just two files: the exe + model.bin.%N")
			io.put_string ("  * Each prompt produces about 400 characters and takes a few seconds on the CPU.%N")
			io.put_string ("  * Built with a hand-written Eiffel inference engine - no Python or GPU needed to run.%N")
			io.put_string ("%N")
		end

end
