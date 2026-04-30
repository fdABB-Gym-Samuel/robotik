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
          python = pkgs.python312;
          py = pkgs.python312Packages;

          # nixpkgs builds opencv4 with GUI disabled by default, so cv2.imshow
          # is a no-op. Enable GTK3 so --display works on Ubuntu/GNOME.
          opencvPy = py.opencv4.override {
            enableGtk3 = true;
            enableContrib = true;
          };

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
            version = "0.10.5";
            pyproject = true;
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/91/cf/28eb9c823dfc245c540f5286d71b44aeee2a51021fc85b25bb9562be78cc/cyclonedds-0.10.5.tar.gz";
              hash = "sha256-Y/xNb9sv01GBxA9OkHVxSfLe9fVw7xn7ce3E9Wh1X4o=";
            };
            build-system = with py; [
              setuptools
              wheel
            ];
            nativeBuildInputs = [ pkgs.pkg-config ];
            buildInputs = [ pkgs.cyclonedds ];
            propagatedBuildInputs = [ richClickPy ];
            env.CYCLONEDDS_HOME = pkgs.cyclonedds;
            pythonImportsCheck = [
              "cyclonedds"
              "cyclonedds.domain"
            ];
            doCheck = false;
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

          mediapipePy = py.buildPythonPackage rec {
            pname = "mediapipe";
            version = "0.10.35";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/32/8f/1bc57dbc9b7b03c8f875aac23380ec57e9002cc02fe6720045fb263f3966/mediapipe-0.10.35-py3-none-manylinux_2_28_x86_64.whl";
              hash = "sha256-25pXnfSM/+lXDNPpP2pdLdCJoRA7hGxgxbXeiiHDjbA=";
            };
            nativeBuildInputs = [ pkgs.autoPatchelfHook ];
            propagatedBuildInputs = [
              py."absl-py"
              py.certifi
              py.flatbuffers
              py.matplotlib
              py.numpy
              opencvPy
              py.protobuf
              py.sounddevice
            ];
            pythonRemoveDeps = [ "opencv-contrib-python" ];
            pythonImportsCheck = [ "mediapipe" ];
            doCheck = false;
          };

          pythonEnv = python.withPackages (
            ps: with ps; [
              cycloneddsPy
              unitreeSdk2Py
              mediapipePy
              numpy
              matplotlib
              mujoco
              opencvPy
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
            mediapipePy
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
              gtk3
              glib
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
