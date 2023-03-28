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
        py = pkgs.python39Packages;

        edit = pkgs.writeScriptBin "edit" ''
            #!/usr/bin/env fish
            set name=case_ai
            zellij a $name || zellij -s $name -l python
            '';

        runTime= [
            py.librosa
            py.pandas
            py.soundfile
            py.numpy
        ];
        devTools = [
          py.venvShellHook
          py.flake8
          py.ipython
          py.black
          edit
        ]; 
        
        
      in
        {
          devShell = pkgs.mkShell  {
            name = "for science";
            venvDir = ".venv";
            root = ./.;

            buildInputs = [
              runTime
              devTools
            ];

            MPLBACKEND = "webagg";
            QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins";

            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
              # allow pip to install wheels
              pip install pytest
              pip install -e .
            '';
            postShellHook = ''
              # allow pip to install wheels
              unset SOURCE_DATE_EPOCH
              # allow pip to install wheels
              export DB_PATH=$PWD/db/
              export IPYTHONDIR=$PWD/.ipy/           
              echo Welcome to the Case event Env!
            '';
          };
        }
    );
}
