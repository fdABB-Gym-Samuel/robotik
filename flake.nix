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

      apps = eachSystem (pkgs:
        let
          richClickPy = pkgs.python313Packages.buildPythonPackage rec {
            pname = "rich-click";
            version = "1.9.7";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/ca/e5/d708d262b600a352abe01c2ae360d8ff75b0af819b78e9af293191d928e6/rich_click-1.9.7-py3-none-any.whl";
              hash = "sha256-L5kSD8p49TbgexFNO2AzO8S7KglpBTsSUIabzcG1NRs=";
            };
            propagatedBuildInputs = with pkgs.python313Packages; [
              click
              rich
            ];
          };
          cycloneddsPy = pkgs.python313Packages.buildPythonPackage rec {
            pname = "cyclonedds";
            version = "11.0.1";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/eb/a5/a6c8052dafd16c8c0e02eb6ef0cb2bb086d726d654c56e94ec4cdb1640ab/cyclonedds-11.0.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
              hash = "sha256-/vH2zhFaV54yLzFWhC6LZgiD5Oiwqi9eKj3515WdFzY=";
            };
            nativeBuildInputs = [ pkgs.autoPatchelfHook ];
            propagatedBuildInputs = [ richClickPy ];
          };
          pythonEnv = pkgs.python313.withPackages (
            ps: with ps; [
              cycloneddsPy
              numpy
              matplotlib
              mujoco
              imageio
              trimesh
              glfw
              typing-extensions
            ]
          );
          g1Demo = pkgs.writeShellApplication {
            name = "g1-rps-demo";
            runtimeInputs = [ pythonEnv ];
            text = ''
              export MUJOCO_GL=glfw
              export LD_LIBRARY_PATH="${pkgs.libGL}/lib:${pkgs.mesa}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              exec ${pythonEnv}/bin/python scripts/run_g1_rps_demo.py "$@"
            '';
          };
          g1HandHardware = pkgs.writeShellApplication {
            name = "g1-rps-hand-hardware";
            runtimeInputs = [ pythonEnv ];
            text = ''
              exec ${pythonEnv}/bin/python scripts/run_g1_rps_hand_hardware.py "$@"
            '';
          };
        in
        {
          default = {
            type = "app";
            program = "${g1Demo}/bin/g1-rps-demo";
          };
          "g1-rps-demo" = {
            type = "app";
            program = "${g1Demo}/bin/g1-rps-demo";
          };
          "g1-rps-hand-hardware" = {
            type = "app";
            program = "${g1HandHardware}/bin/g1-rps-hand-hardware";
          };
        });

      devShells = eachSystem (pkgs: {
        default =
          let
            richClickPy = pkgs.python313Packages.buildPythonPackage rec {
              pname = "rich-click";
              version = "1.9.7";
              format = "wheel";
              src = pkgs.fetchurl {
                url = "https://files.pythonhosted.org/packages/ca/e5/d708d262b600a352abe01c2ae360d8ff75b0af819b78e9af293191d928e6/rich_click-1.9.7-py3-none-any.whl";
                hash = "sha256-L5kSD8p49TbgexFNO2AzO8S7KglpBTsSUIabzcG1NRs=";
              };
              propagatedBuildInputs = with pkgs.python313Packages; [
                click
                rich
              ];
            };
            cycloneddsPy = pkgs.python313Packages.buildPythonPackage rec {
              pname = "cyclonedds";
              version = "11.0.1";
              format = "wheel";
              src = pkgs.fetchurl {
                url = "https://files.pythonhosted.org/packages/eb/a5/a6c8052dafd16c8c0e02eb6ef0cb2bb086d726d654c56e94ec4cdb1640ab/cyclonedds-11.0.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
                hash = "sha256-/vH2zhFaV54yLzFWhC6LZgiD5Oiwqi9eKj3515WdFzY=";
              };
              nativeBuildInputs = [ pkgs.autoPatchelfHook ];
              propagatedBuildInputs = [ richClickPy ];
            };
            pythonEnv = pkgs.python313.withPackages (
              ps: with ps; [
                cycloneddsPy
                numpy
                matplotlib
                mujoco
                imageio
                trimesh
                glfw
                typing-extensions
              ]
            );
            g1Demo = pkgs.writeShellApplication {
              name = "g1-rps-demo";
              runtimeInputs = [ pythonEnv ];
              text = ''
                export MUJOCO_GL=glfw
                export LD_LIBRARY_PATH="${pkgs.libGL}/lib:${pkgs.mesa}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                exec ${pythonEnv}/bin/python scripts/run_g1_rps_demo.py "$@"
              '';
            };
            g1HandHardware = pkgs.writeShellApplication {
              name = "g1-rps-hand-hardware";
              runtimeInputs = [ pythonEnv ];
              text = ''
                exec ${pythonEnv}/bin/python scripts/run_g1_rps_hand_hardware.py "$@"
              '';
            };
          in
          pkgs.mkShell {
          packages = with pkgs; [
            curl
            git
            nixfmt
            pkg-config
            libGL
            libGLU
            mesa
            libX11
            libXext
            pythonEnv
            g1Demo
            g1HandHardware
          ];
          env = {
            MUJOCO_GL = "glfw";
            LD_LIBRARY_PATH = "${pkgs.libGL}/lib:${pkgs.mesa}/lib";
          };
        };
      });
    };
}
