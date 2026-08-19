# cherimoya_cli install-skill command
# Author: The Cherimoya Authors


def add_parser(subparsers):
	parser = subparsers.add_parser("install-skill",
		help="Install the bundled Cherimoya agent skill for Claude Code.")
	parser.add_argument("-d", "--directory", type=str, default=None,
		help="Skills directory to install into. Default is "
		"~/.claude/skills.")
	parser.add_argument("--symlink", action='store_true', default=False,
		help="Symlink the packaged skill instead of copying it. Reflects "
		"in-place edits, but breaks if the install location moves.")
	parser.add_argument("-f", "--force", action='store_true', default=False,
		help="Overwrite an existing installation at the destination.")
	return parser


def run(args):
	import os
	import shutil

	source = os.path.join(os.path.dirname(os.path.dirname(
		os.path.abspath(__file__))), "skills", "cherimoya")

	if not os.path.isdir(source):
		raise FileNotFoundError(
			"Bundled skill not found at {}; the package may be installed "
			"without its data files.".format(source))

	if args.directory is not None:
		skills_dir = os.path.expanduser(args.directory)
	else:
		skills_dir = os.path.expanduser(os.path.join("~", ".claude", "skills"))

	os.makedirs(skills_dir, exist_ok=True)
	dest = os.path.join(skills_dir, "cherimoya")

	if os.path.lexists(dest):
		if not args.force:
			raise FileExistsError(
				"A skill already exists at {}. Re-run with --force to "
				"overwrite it.".format(dest))

		if os.path.islink(dest) or os.path.isfile(dest):
			os.remove(dest)
		else:
			shutil.rmtree(dest)

	if args.symlink:
		os.symlink(source, dest)
		print("Symlinked Cherimoya skill:\n  {} -> {}".format(dest, source))
	else:
		shutil.copytree(source, dest,
			ignore=shutil.ignore_patterns(".ipynb_checkpoints"))
		print("Installed Cherimoya skill to:\n  {}".format(dest))

	print("Restart Claude Code (or reload skills) to pick it up.")
