{
  inputs =  {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    mach-nix.url = "mach-nix/3.5.0";
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix}:

    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit  system;
        };
        mach = mach-nix.lib.${system};
        pythonVersion = "python39";
        pythonEnv = mach.mkPython {
          python = pythonVersion;
          # requirments = builtins.readFile ./requirements.txt;
          requirements = ''
            numpy
            scipy
            matplotlib
            ipython
            keras
            tensorflow
            librosa
            pandas
            soundfile
            # flask
            # waitress
          '';
          providers.soundfile = "nixpkgs";
            
        };


      in
        {
          devShell = pkgs.mkShell  {
            name = "for science";
            venvDir = ".venv";
            root = ./.;

            buildInputs = [
              pythonEnv
            ];

            MPLBACKEND = "webagg";
            
            QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins";
            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
            '';

            shellHook = ''
              # allow pip to install wheels
              export DB_PATH=$PWD/db/
              echo Welcome to the Case event Env!
              unset SOURCE_DATE_EPOCH
            '';
          };
        }
    );
}
