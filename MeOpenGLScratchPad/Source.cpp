#include <qtgui\qapplication.h>
#include <Rendering\Renderer.h>
#include <IO\FileIO.h>
#include <Rendering\Helpers\Renderables.h>
#include <MyGame.h>
#include "Asteroids\AsterMain.h"
#include "Other_Tests\RBM.h"

bool Test_Asteroids = true;
bool Testing_Others = true;

void main(int argumentCount, char* argumentVector[]){
	if (Testing_Others){
		RBM* rbm = new RBM(10, 5, 5, NULL, NULL, NULL);
		rbm->test_rbm();
	}
	else if (Test_Asteroids){
		AsterMain newGame;
		newGame.run();
	}
	
	else{
		QApplication app(argumentCount, argumentVector);
		MyGame game;

		if (!game.initialize())
			return;

		game.gameLoop();

		app.exec();
		game.shutdown();
	};
}
