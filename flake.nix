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

      apps = eachSystem (
        pkgs:
        let
          richClickPy = pkgs.python312Packages.buildPythonPackage rec {
            pname = "rich-click";
            version = "1.9.7";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/ca/e5/d708d262b600a352abe01c2ae360d8ff75b0af819b78e9af293191d928e6/rich_click-1.9.7-py3-none-any.whl";
              hash = "sha256-L5kSD8p49TbgexFNO2AzO8S7KglpBTsSUIabzcG1NRs=";
            };
            propagatedBuildInputs = with pkgs.python312Packages; [
              click
              rich
            ];
          };
          cycloneddsPy = pkgs.python312Packages.buildPythonPackage rec {
            pname = "cyclonedds";
            version = "11.0.1";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/5a/27/26dafd6cde19a440497c26d3fd39560db2e5ec2261fa628801000a0cd8b6/cyclonedds-11.0.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
              hash = "sha256-PpZQcIjFcWX3wYnDqFvoZvdMdEn7DLxjFq0wbl9Zm+E=";
            };
            nativeBuildInputs = [ pkgs.autoPatchelfHook ];
            propagatedBuildInputs = [ richClickPy ];
          };
          mediapipePy = pkgs.python312Packages.buildPythonPackage rec {
            pname = "mediapipe";
            version = "0.10.30";
            format = "wheel";
            # 0.10.30 ships a ``py3-none`` wheel with the native libraries
            # decoupled from the CPython ABI, which sidesteps the numpy 1
            # vs numpy 2 ABI segfault we hit with 0.10.21 on numpy 2.x.
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/34/cf/fb95dfebccace031576dc4d7c91b257aec05f0dd33cf63de59275e314e3e/mediapipe-0.10.30-py3-none-manylinux_2_28_x86_64.whl";
              hash = "sha256-Uo17sTpQk/3hjncUEGN5R+0jLBKDTOyKQfhHSpvyC/A=";
            };
            nativeBuildInputs = [ pkgs.autoPatchelfHook ];
            propagatedBuildInputs = with pkgs.python312Packages; [
              absl-py
              flatbuffers
              numpy
              sounddevice
            ];
          };
          pythonEnv = pkgs.python312.withPackages (
            ps: with ps; [
              cycloneddsPy
              numpy
              matplotlib
              mujoco
              imageio
              trimesh
              glfw
              typing-extensions
              (opencv4.override { enableGtk3 = true; })
              mediapipePy
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
          g1VisionDemo = pkgs.writeShellApplication {
            name = "g1-rps-vision-demo";
            runtimeInputs = [ pythonEnv ];
            text = ''
              export LD_LIBRARY_PATH="${pkgs.libGL}/lib:${pkgs.mesa}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              exec ${pythonEnv}/bin/python scripts/run_g1_rps_vision_demo.py "$@"
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
          "g1-rps-vision-demo" = {
            type = "app";
            program = "${g1VisionDemo}/bin/g1-rps-vision-demo";
          };
        }
      );

      devShells = eachSystem (pkgs: {
        default =
          let
            richClickPy = pkgs.python312Packages.buildPythonPackage rec {
              pname = "rich-click";
              version = "1.9.7";
              format = "wheel";
              src = pkgs.fetchurl {
                url = "https://files.pythonhosted.org/packages/ca/e5/d708d262b600a352abe01c2ae360d8ff75b0af819b78e9af293191d928e6/rich_click-1.9.7-py3-none-any.whl";
                hash = "sha256-L5kSD8p49TbgexFNO2AzO8S7KglpBTsSUIabzcG1NRs=";
              };
              propagatedBuildInputs = with pkgs.python312Packages; [
                click
                rich
              ];
            };
            cycloneddsPy = pkgs.python312Packages.buildPythonPackage rec {
              pname = "cyclonedds";
              version = "11.0.1";
              format = "wheel";
              src = pkgs.fetchurl {
                url = "https://files.pythonhosted.org/packages/5a/27/26dafd6cde19a440497c26d3fd39560db2e5ec2261fa628801000a0cd8b6/cyclonedds-11.0.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
                hash = "sha256-PpZQcIjFcWX3wYnDqFvoZvdMdEn7DLxjFq0wbl9Zm+E=";
              };
              nativeBuildInputs = [ pkgs.autoPatchelfHook ];
              propagatedBuildInputs = [ richClickPy ];
            };
            mediapipePy = pkgs.python312Packages.buildPythonPackage rec {
              pname = "mediapipe";
              version = "0.10.30";
              format = "wheel";
              # 0.10.30 ships a ``py3-none`` wheel with the native libraries
              # decoupled from the CPython ABI, which sidesteps the numpy 1
              # vs numpy 2 ABI segfault we hit with 0.10.21 on numpy 2.x.
              src = pkgs.fetchurl {
                url = "https://files.pythonhosted.org/packages/34/cf/fb95dfebccace031576dc4d7c91b257aec05f0dd33cf63de59275e314e3e/mediapipe-0.10.30-py3-none-manylinux_2_28_x86_64.whl";
                hash = "sha256-Uo17sTpQk/3hjncUEGN5R+0jLBKDTOyKQfhHSpvyC/A=";
              };
              nativeBuildInputs = [ pkgs.autoPatchelfHook ];
              propagatedBuildInputs = with pkgs.python312Packages; [
                absl-py
                flatbuffers
                numpy
                sounddevice
              ];
            };
            pythonEnv = pkgs.python312.withPackages (
              ps: with ps; [
                cycloneddsPy
                numpy
                matplotlib
                mujoco
                imageio
                trimesh
                glfw
                typing-extensions
                (opencv4.override { enableGtk3 = true; })
                mediapipePy
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
            g1VisionDemo = pkgs.writeShellApplication {
              name = "g1-rps-vision-demo";
              runtimeInputs = [ pythonEnv ];
              text = ''
                export LD_LIBRARY_PATH="${pkgs.libGL}/lib:${pkgs.mesa}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                exec ${pythonEnv}/bin/python scripts/run_g1_rps_vision_demo.py "$@"
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
              g1VisionDemo
            ];
            env = {
              MUJOCO_GL = "glfw";
              LD_LIBRARY_PATH = "${pkgs.libGL}/lib:${pkgs.mesa}/lib";
            };
          };
      });
    };
}
