{ pkgs, ... }:
let

in
{
  projectRootFile = "flake.nix";

  programs = {
    dos2unix.enable = true;
    mdformat.enable = true;
    nixfmt.enable = true;
    yamlfmt.enable = true;
    ruff-format.enable = true;
  };

}
