import gpudecide
import argparse

parser = argparse.ArgumentParser(description="Generate the BDD for the N-queens puzzle")
parser.add_argument("-n", type=int, default=6)

parser.add_argument("-C", type=int, default=10000)
parser.add_argument("-G", type=int, default=30000)


args = parser.parse_args()

def at_most_one(vars, bdd):
	assert(len(vars) >= 2)

	has_true_occured_once = bdd.logical_or(vars[0], vars[1])
	has_true_occured_less_than_twice = bdd.logical_nand(vars[0], vars[1])

	for i, v in enumerate(vars[2:]):
		has_true_occured_less_than_twice = bdd.logical_and(bdd.logical_nand(has_true_occured_once, v), has_true_occured_less_than_twice)
		if i < len(vars) - 3:
			has_true_occured_once = bdd.logical_or(has_true_occured_once, v)
	return has_true_occured_less_than_twice

def exactly_one(vars, bdd):
	assert(len(vars) >= 2)

	has_true_occured_once = bdd.logical_or(vars[0], vars[1])
	has_true_occured_less_than_twice = bdd.logical_nand(vars[0], vars[1])

	for i, v in enumerate(vars[2:]):
		has_true_occured_less_than_twice = bdd.logical_and(bdd.logical_nand(has_true_occured_once, v), has_true_occured_less_than_twice)
		has_true_occured_once = bdd.logical_or(has_true_occured_once, v)
	return bdd.logical_and(has_true_occured_once, has_true_occured_less_than_twice)

def flatten_xy(n, x, y):
	return x + n * y

def diagonal(n, board, index, up):
	x = 0
	y = index

	while y >= n:
		x += 1
		y -= 1
	
	diagonal = []
	while x < n and y >= 0:
		diagonal.append(board[flatten_xy(n, x if up else n-x-1, y)])
		x += 1
		y -= 1
	
	return at_most_one(diagonal, board)

def test_n_queens(n):
	admin = gpudecide.thrust_administrator()
	admin.switch_to_multi_cpu_at = args.C
	admin.switch_to_gpu_at = args.G

	board = gpudecide.BDD(n * n, admin)

	statements = []
	
	# Unique queen in columns
	for i in range(n):
		row = []
		column = []
		for j in range(n):
			row.append(board[flatten_xy(n, i, j)])
			column.append(board[flatten_xy(n, j, i)])
		
		statements.append(at_most_one(row, board))
		statements.append(exactly_one(column, board))
	
	# Diagonals
	for i in range(1, 2*n-2):
		statements.append(diagonal(n, board, i, False))
		statements.append(diagonal(n, board, i, True))
	
	result = board.logical_and(statements[0], statements[1])
	for s in statements[2:]:
		# print("Node count: ", board.count_nodes(), board.count_irreducible_nodes())
		result = board.logical_and(s, result)

	return board, result 

n = args.n
board, result = test_n_queens(n)

print("Satisfying assignment count: ", result.get_metric(board.count_satisfying_assignments()))
print("Total node count: ", board.count_nodes())
print("Irreducible node count: ", board.count_irreducible_nodes())

# Print all configurations
#for p in result:
#	for i in range(0, n * n, n):
#		print("".join(["*" if p[j] else "-" for j in range(i, i + n)]))
#	
#	print()