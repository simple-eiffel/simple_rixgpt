note
	description: "Test runner for simple_rixgpt_train (M2)."
	author: "Larry Rix"

class
	TRAIN_TEST_APP

create
	make

feature {NONE} -- Initialization

	make
		local
			t: TRAIN_TESTS
		do
			create t
			print ("simple_rixgpt training tests%N============================%N")
			run (agent t.test_rixg_dimensions, "model.bin dimensions and weight count")
			run (agent t.test_corpus_matches_recipe, "corpus vocabulary = rixgpt.py's (model.bin)")
			run (agent t.test_parity_oracle, "trained weights: logits = RIXGPT_MODEL (oracle)")
			run (agent t.test_parity_gpu, "trained weights: logits = RIXGPT_MODEL (GPU)")
			run (agent t.test_tiny_overfit_gpu, "tiny RixGPT overfits a short text (GPU)")
			print ("============================%NResults: " + passed.out + " passed, " + failed.out + " failed%N")
		end

feature {NONE} -- Runner

	passed, failed: INTEGER

	run (a_test: PROCEDURE; a_name: STRING)
		local
			l_retried: BOOLEAN
		do
			if not l_retried then
				a_test.call (Void)
				print ("  PASS: " + a_name + "%N")
				passed := passed + 1
			end
		rescue
			print ("  FAIL: " + a_name + "%N")
			if attached (create {EXCEPTION_MANAGER}).last_exception as e then
				print ("        " + e.generating_type.name + " in " + if attached e.type_name as tn then tn.to_string_8 else "?" end + "." + if attached e.recipient_name as rn then rn.to_string_8 else "?" end + ": " + if attached e.description as d then d.to_string_8 else "" end + "%N")
			end
			failed := failed + 1
			l_retried := True
			retry
		end

end
