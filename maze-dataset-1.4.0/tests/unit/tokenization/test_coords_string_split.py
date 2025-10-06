from maze_dataset.token_utils import coords_string_split_UT


def test_coords_string_split_UT():
	assert coords_string_split_UT("(1,2) (3,4)") == ["(1,2)", "(3,4)"]
	assert coords_string_split_UT("(1,2)(3,4)") == ["(1,2)", "(3,4)"]
	assert coords_string_split_UT("(1,2) (3,4) (5,6)") == ["(1,2)", "(3,4)", "(5,6)"]
	assert coords_string_split_UT("()") == ["()"]
	assert coords_string_split_UT("(,)(,)") == ["(,)", "(,)"]
	assert coords_string_split_UT("( , ) ( , )") == ["( , )", "( , )"]
	assert coords_string_split_UT("(1,2) (3,4) (5,6) (7,8)") == [
		"(1,2)",
		"(3,4)",
		"(5,6)",
		"(7,8)",
	]
	assert coords_string_split_UT("") == []
	assert coords_string_split_UT("(1, 2) (3, 4)") == ["(1, 2)", "(3, 4)"]
	assert coords_string_split_UT("(1 ,2) (3,4)") == ["(1 ,2)", "(3,4)"]
	assert coords_string_split_UT("(1,2) (3 ,4)") == ["(1,2)", "(3 ,4)"]
	assert coords_string_split_UT(" ( 1 , 2 ) ( 3 , 4 )") == ["( 1 , 2 )", "( 3 , 4 )"]
	assert coords_string_split_UT("(1,2) (3, 4 )") == ["(1,2)", "(3, 4 )"]
	assert coords_string_split_UT("(1 , 2) (3 , 4)") == ["(1 , 2)", "(3 , 4)"]
	assert coords_string_split_UT("(1,2) 3,4 (5,6)") == ["(1,2)", "3,4", "(5,6)"]
	assert coords_string_split_UT("(1,2) 3 , 4 (5,6)") == [
		"(1,2)",
		"3",
		",",
		"4",
		"(5,6)",
	]
	assert coords_string_split_UT("(1,2) <SPECIAL_TOKEN> (3,4)") == [
		"(1,2)",
		"<SPECIAL_TOKEN>",
		"(3,4)",
	]
	assert coords_string_split_UT("<SPECIAL_TOKEN> (1,2) (3,4)") == [
		"<SPECIAL_TOKEN>",
		"(1,2)",
		"(3,4)",
	]
	assert coords_string_split_UT("(1,2) (3,4) <SPECIAL_TOKEN>") == [
		"(1,2)",
		"(3,4)",
		"<SPECIAL_TOKEN>",
	]
	assert coords_string_split_UT("<SPECIAL_TOKEN>") == ["<SPECIAL_TOKEN>"]
	assert coords_string_split_UT("(1,2) <SPECIAL_TOKEN> (3,4) <SPECIAL_TOKEN>") == [
		"(1,2)",
		"<SPECIAL_TOKEN>",
		"(3,4)",
		"<SPECIAL_TOKEN>",
	]
	assert coords_string_split_UT(
		" ( 1 , 2 ) <SPECIAL_TOKEN> ( 3 , 4 ) <SPECIAL_TOKEN>",
	) == [
		"( 1 , 2 )",
		"<SPECIAL_TOKEN>",
		"( 3 , 4 )",
		"<SPECIAL_TOKEN>",
	]
	assert coords_string_split_UT(
		"<SPECIAL_TOKEN> (1,2) <SPECIAL_TOKEN> (3,4) <SPECIAL_TOKEN>",
	) == ["<SPECIAL_TOKEN>", "(1,2)", "<SPECIAL_TOKEN>", "(3,4)", "<SPECIAL_TOKEN>"]
	assert coords_string_split_UT(
		"(1,2) <SPECIAL_TOKEN> (3,4) <SPECIAL_TOKEN> (5,6)",
	) == [
		"(1,2)",
		"<SPECIAL_TOKEN>",
		"(3,4)",
		"<SPECIAL_TOKEN>",
		"(5,6)",
	]
	assert coords_string_split_UT(
		"<SPECIAL_TOKEN> <SPECIAL_TOKEN> <SPECIAL_TOKEN>",
	) == [
		"<SPECIAL_TOKEN>",
		"<SPECIAL_TOKEN>",
		"<SPECIAL_TOKEN>",
	]
	assert coords_string_split_UT("1 2 3") == ["1", "2", "3"]
	assert coords_string_split_UT("(1,2) 3 (5,6)") == ["(1,2)", "3", "(5,6)"]
	assert coords_string_split_UT("( 1 , 2 ) 3 ( 5 , 6 )") == [
		"( 1 , 2 )",
		"3",
		"( 5 , 6 )",
	]
	assert coords_string_split_UT("1 <SPECIAL_TOKEN> 2") == [
		"1",
		"<SPECIAL_TOKEN>",
		"2",
	]
	assert coords_string_split_UT("<SPECIAL_TOKEN> 1 <SPECIAL_TOKEN>") == [
		"<SPECIAL_TOKEN>",
		"1",
		"<SPECIAL_TOKEN>",
	]
	assert coords_string_split_UT("1 <SPECIAL_TOKEN> 2 <SPECIAL_TOKEN>") == [
		"1",
		"<SPECIAL_TOKEN>",
		"2",
		"<SPECIAL_TOKEN>",
	]
	assert coords_string_split_UT(
		"<SPECIAL_TOKEN> <SPECIAL_TOKEN> <SPECIAL_TOKEN>",
	) == [
		"<SPECIAL_TOKEN>",
		"<SPECIAL_TOKEN>",
		"<SPECIAL_TOKEN>",
	]
	assert coords_string_split_UT(
		"(1,2) <SPECIAL_TOKEN> (3,4) 7 <SPECIAL_TOKEN> (5,6)",
	) == ["(1,2)", "<SPECIAL_TOKEN>", "(3,4)", "7", "<SPECIAL_TOKEN>", "(5,6)"]
	assert coords_string_split_UT(
		" ( 1 , 2 ) <SPECIAL_TOKEN> ( 3 , 4 ) 7 <SPECIAL_TOKEN> ( 5 , 6 )",
	) == [
		"( 1 , 2 )",
		"<SPECIAL_TOKEN>",
		"( 3 , 4 )",
		"7",
		"<SPECIAL_TOKEN>",
		"( 5 , 6 )",
	]
