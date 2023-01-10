name = topic_list
slides = slides_test
poster = poster_test
debug = debug


topic_list.pdf:
	lualatex -interaction=nonstopmode -halt-on-error -shell-escape $(name)
	biber $(name)
	lualatex -interaction=nonstopmode $(name)

slides_test.pdf:
	lualatex -interaction=nonstopmode -halt-on-error -shell-escape $(slides)
	biber $(slides)
	lualatex $(slides)
	

poster_test.pdf:
	lualatex -interaction=nonstopmode -halt-on-error -shell-escape $(poster)
	biber $(poster)
	lualatex $(poster)

debug.pdf:
	lualatex -interaction=nonstopmode -halt-on-error -shell-escape $(debug)
	biber $(debug)
	lualatex $(debug)
