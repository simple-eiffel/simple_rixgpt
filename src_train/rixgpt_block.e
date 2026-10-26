note
	description: "[
		One RixGPT transformer block (rixgpt.py Block): pre-norm attention and MLP with
		residuals. x := x + attn (ln1 (x)); x := x + mlp (ln2 (x)). Children named as in
		PyTorch: ln1, attn, ln2, mlp.
	]"
	author: "Larry Rix"

class
	RIXGPT_BLOCK

inherit
	NN_LAYER
		redefine
			accepts
		end

create
	make

feature {NONE} -- Initialization

	make (f: SIMPLE_TENSOR; c: RIXGPT_CONFIG; r: TENSOR_RANDOM)
		do
			make_module
			create ln1.make (f, c.n_embd, 1.0e-5)
			create attn.make (f, c.n_embd, c.n_head, c.dropout, 0.02, r)
			create ln2.make (f, c.n_embd, 1.0e-5)
			create mlp.make (f, c.n_embd, 4 * c.n_embd, c.dropout, 0.02, r)
			register_child ("ln1", ln1)
			register_child ("attn", attn)
			register_child ("ln2", ln2)
			register_child ("mlp", mlp)
		end

feature -- Access

	ln1, ln2: NN_LAYER_NORM
	attn: NN_CAUSAL_SELF_ATTENTION
	mlp: NN_MLP

feature -- Status report

	accepts (x: TENSOR): BOOLEAN
		do
			Result := attn.accepts (x)
		end

feature -- Basic operations

	forward (x: TENSOR): TENSOR
		do
			Result := x + attn.forward (ln1.forward (x))
			Result := Result + mlp.forward (ln2.forward (Result))
		ensure then
			same_shape: Result.shape ~ x.shape
		end

end
