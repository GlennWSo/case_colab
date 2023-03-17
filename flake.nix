{
  inputs =  {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pyvista = {
      type = "github";
      owner = "GlennWSo";
      repo = "pyvista";
      rev = "d1b5e66928fb3c85d449fd44a04f74139e43d1d9";
    };
  };

  outputs = { self, nixpkgs, flake-utils, pyvista}:

    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit  system;
        };
        py = pkgs.python310Packages;
        pv = pyvista.packages.${system}.pyvista;


      in
        {
          devShell = pkgs.mkShell  {
            name = "flake pyrust";
            venvDir = ".venv";
            root = ./.;

            buildInputs = [
              pv
              py.venvShellHook
              py.black
              py.pandas
              py.numpy
              py.matplotlib
            ];

            MPLBACKEND = "webagg";
            # QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins";
            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
              pip install ipython
              pip install rscad
            '';

            postShellHook = ''
              # allow pip to install wheels
              echo Welcome to the Case event Env!
              unset SOURCE_DATE_EPOCH
            '';
          };
        }
    );
}
