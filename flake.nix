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

      mkProjectPackages =
        pkgs:
        let
          python = pkgs.python313;
          py = pkgs.python313Packages;

          richClickPy = py.buildPythonPackage rec {
            pname = "rich-click";
            version = "1.9.7";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/ca/e5/d708d262b600a352abe01c2ae360d8ff75b0af819b78e9af293191d928e6/rich_click-1.9.7-py3-none-any.whl";
              hash = "sha256-L5kSD8p49TbgexFNO2AzO8S7KglpBTsSUIabzcG1NRs=";
            };
            propagatedBuildInputs = with py; [
              click
              rich
            ];
          };

          cycloneddsPy = py.buildPythonPackage rec {
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

          unitreeSdk2Py = py.buildPythonPackage rec {
            pname = "unitree-sdk2py";
            version = "1.0.1";
            pyproject = true;
            src = pkgs.fetchFromGitHub {
              owner = "unitreerobotics";
              repo = "unitree_sdk2_python";
              rev = "db9b2d210081387fcd1e7ed9ac4c56a02983bb85";
              hash = "sha256-doQBu8Ctnrt7CCQjtByWMkf1SIWXAbV/ik05hmCsHj4=";
            };
            build-system = with py; [
              setuptools
              wheel
            ];
            postPatch = ''
              substituteInPlace setup.py \
                --replace-fail '"cyclonedds==0.10.2",' "" \
                --replace-fail '"opencv-python",' ""
              cat > unitree_sdk2py/__init__.py <<'EOF'
              __all__ = [
                "idl",
                "utils",
                "core",
                "rpc",
                "comm",
                "g1",
                "go2",
                "h1",
                "b2",
              ]
              EOF
            '';
            postInstall = ''
              mkdir -p $out/${python.sitePackages}/unitree_sdk2py/utils
              cp -r unitree_sdk2py/utils/lib $out/${python.sitePackages}/unitree_sdk2py/utils/
            '';
            propagatedBuildInputs = [
              cycloneddsPy
              py.numpy
            ];
            pythonImportsCheck = [ "unitree_sdk2py.core.channel" ];
            doCheck = false;
          };

          pythonEnv = python.withPackages (
            ps: with ps; [
              cycloneddsPy
              unitreeSdk2Py
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

          g1ArmHardware = pkgs.writeShellApplication {
            name = "g1-pre-reveal-right-arm-hardware";
            runtimeInputs = [ pythonEnv ];
            text = ''
              exec ${pythonEnv}/bin/python scripts/pre_reveal_right_arm_hardware.py "$@"
            '';
          };
        in
        {
          inherit
            cycloneddsPy
            g1ArmHardware
            g1Demo
            g1HandHardware
            pythonEnv
            richClickPy
            unitreeSdk2Py
            ;
        };
    in
    {
      formatter = eachSystem (pkgs: treefmtEval.${pkgs.stdenv.hostPlatform.system}.config.build.wrapper);

      apps = eachSystem (
        pkgs:
        let
          project = mkProjectPackages pkgs;
        in
        {
          default = {
            type = "app";
            program = "${project.g1Demo}/bin/g1-rps-demo";
          };
          "g1-rps-demo" = {
            type = "app";
            program = "${project.g1Demo}/bin/g1-rps-demo";
          };
          "g1-rps-hand-hardware" = {
            type = "app";
            program = "${project.g1HandHardware}/bin/g1-rps-hand-hardware";
          };
          "g1-pre-reveal-right-arm-hardware" = {
            type = "app";
            program = "${project.g1ArmHardware}/bin/g1-pre-reveal-right-arm-hardware";
          };
        }
      );

      devShells = eachSystem (
        pkgs:
        let
          project = mkProjectPackages pkgs;
        in
        {
          default = pkgs.mkShell {
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
              project.pythonEnv
              project.g1Demo
              project.g1HandHardware
              project.g1ArmHardware
            ];
            env = {
              MUJOCO_GL = "glfw";
              LD_LIBRARY_PATH = "${pkgs.libGL}/lib:${pkgs.mesa}/lib";
            };
          };
        }
      );
    };
}
