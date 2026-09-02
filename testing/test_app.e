note
	description: "[
		Compile proof for simple_rixgpt.

		This is not behavioral coverage and does not pretend to be.
		RIXGPT_MODEL has exactly one creation procedure, `make (a_path)', and it
		takes the path of a RIXG weights file. The weights are not in this
		repository - they are a build payload from a private training pipeline -
		so nothing in the class can be exercised here without inventing a
		fixture, and a fabricated model would prove nothing about the real one.

		What this target does prove is that RIXGPT_MODEL and RIXGPT_APP both
		type-check and finalize. Declaring a local of a class puts that class in
		the system, so the whole of each is compiled. That is the check that
		breaks first when the engine is edited, and it is the acceptance this
		repository can honestly offer.
	]"
	author: "Larry Rix"

class
	TEST_APP

create
	make

feature {NONE} -- Initialization

	make
			-- Report that the model and app classes are in the compiled system.
		local
			l_model: detachable RIXGPT_MODEL
			l_app: detachable RIXGPT_APP
		do
			print ("Running simple_rixgpt tests...%N%N")
			check_that ("RIXGPT_MODEL compiles into the system", not attached l_model)
			check_that ("RIXGPT_APP compiles into the system", not attached l_app)
			print ("%N========================%N")
			print ("Results: " + passed.out + " passed, " + failed.out + " failed%N")
			if failed > 0 then
				print ("TESTS FAILED%N")
			else
				print ("ALL TESTS PASSED%N")
				print ("(compile proof only - generation needs model.bin, which is not in this repository)%N")
			end
		end

feature {NONE} -- State

	passed: INTEGER
			-- Number of checks that held.

	failed: INTEGER
			-- Number of checks that did not.

feature {NONE} -- Assertions

	check_that (a_name: STRING; a_condition: BOOLEAN)
			-- Record a pass/fail line for `a_name'.
		require
			name_not_empty: not a_name.is_empty
		do
			if a_condition then
				print ("  PASS: " + a_name + "%N")
				passed := passed + 1
			else
				print ("  FAIL: " + a_name + "%N")
				failed := failed + 1
			end
		ensure
			one_more_result: passed + failed = old passed + old failed + 1
		end

end
