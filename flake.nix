{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    treefmt-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:numtide/treefmt-nix";
    };
  };
  outputs =
    {
      self,
      nixpkgs,
      systems,
      treefmt-nix,
      ...
    }:
    let
      eachSystem = f: nixpkgs.lib.genAttrs (import systems) (system: f nixpkgs.legacyPackages.${system});
      treefmtEval = eachSystem (pkgs: treefmt-nix.lib.evalModule pkgs ./treefmt.nix);
      version = if (self ? rev) then self.rev else "dirty";
    in
    {
      formatter = eachSystem (pkgs: treefmtEval.${pkgs.stdenv.hostPlatform.system}.config.build.wrapper);

      devShells = eachSystem (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            curl
            git
            nixfmt
            libGL
            libGLU
            mesa
            libX11
            libXext
            (python3.withPackages (
              ps: with ps; [
                numpy
                torch
                matplotlib
                stable-baselines3
                gymnasium
                mujoco
                imageio
              ]
            ))
          ];
          env = {
            MUJOCO_GL = "egl";
            LD_LIBRARY_PATH = "${pkgs.libGL}/lib:${pkgs.mesa}/lib";
          };
        };
      });
    };
}
