#include <gtkmm.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "../Ringbuffer.hpp"
#include "../DataReader.hpp"
#include "FitParser.hpp"

class MainWindow: public Gtk::Window {
private:
    Glib::RefPtr<Gtk::Builder> _builder;
    Gtk::Button *button1,*button2, *button3, *button4;
    Gtk::Image *image;
    
    InputBuffer waveBuffer;
	Fits fits;
    DataReader reader;
    int chunkIndex;
    int waveIndex;
    void drawWaveform() {
    	std::fstream waveFile("wave.txt", std::ios::out);
    	for(int i = 0; i < SAMPLE_COUNT; i++) {
    		waveFile << waveBuffer.getBuffer()[chunkIndex][waveIndex*SAMPLE_COUNT+i] << std::endl;
    	}
    	waveFile.close();
    	const int offset = fits.fits[chunkIndex*CHUNK_COUNT+waveIndex].woffset;
    	std::fstream plotFile("plot.gnu", std::ios::out);
    	plotFile << "set terminal png size 1920,1000" << std::endl
				 << "set output \"plot.png\"" << std::endl
				 << "set xlabel \"time\"" << std::endl
				 << "set ylabel \"energy\"" << std::endl
 				 << "set yrange [-32000: 32000]" << std::endl
			     << "set xrange [0: 1000]" << std::endl
 			     << "set x2range [0: 1000]" << std::endl
 			     << "set arrow from " << offset << "," << -32000 << " to "<< offset << "," << 32000 <<" nohead lc rgb \'black\'" << std::endl
 			     << "set arrow from " << offset+50 << "," << -32000 << " to "<< offset+50 << "," << 32000 <<" nohead lc rgb \'blue\'" << std::endl
 			     << "set arrow from " << offset+100 << "," << -32000 << " to "<< offset+100 << "," << 32000 <<" nohead lc rgb \'black\'" << std::endl
				 << "f(x) = ";
				/*
				 << "f(x) = " << fits.fits[chunkIndex*CHUNK_COUNT+waveIndex].param[0] << "*exp(-1*((x-" << fits.fits[chunkIndex*CHUNK_COUNT+waveIndex].param[1] << ")/" << fits.fits[chunkIndex*CHUNK_COUNT+waveIndex].param[3] << ")**2) + " << fits.fits[chunkIndex*CHUNK_COUNT+waveIndex].param[2];
		*/
		
		for(int i = 0; i < FitFunction::numberOfParams; i++) {
			plotFile << fits.fits[chunkIndex*CHUNK_COUNT+waveIndex].param[i] << "*x**" << i;
			if(i < FitFunction::numberOfParams-1) plotFile << " + ";
		}
		
		plotFile << std::endl;
		std::string color;
		if(fits.fits[chunkIndex*CHUNK_COUNT+waveIndex].status == 0) color = "green";
		else color = "red";
		plotFile << "plot 'wave.txt' using 1 title \"Raw Data (smoothed)\" with line smooth sbezier axes x1y1 lt rgb \"black\", \\" << std::endl
				 << "f(x) with line title \"Fit\" axes x2y1 lt rgb \"" << color << "\"" << std::endl;
		plotFile.close();
    	std::system("gnuplot plot.gnu");
		image->set("plot.png");
    }
public:
	/** "quit" action handler. */
	void OnQuit() {
		hide();
	}
	
	void Button_1_Click() {
		std::cout << "Button " << 1 << " clicked." << std::endl;
		if(chunkIndex > 0) chunkIndex -= 1;
		else waveIndex = 0;
		drawWaveform();

	}
	void Button_2_Click() {
		std::cout << "Button " << 2 << " clicked." << std::endl;
		if(waveIndex > 0) {
			waveIndex -= 1;		
			drawWaveform();
		} else {
			waveIndex = CHUNK_COUNT - 1;
			Button_1_Click();
		}
	}
	void Button_3_Click() {
		std::cout << "Button " << 3 << " clicked." << std::endl;
		if(waveIndex <= CHUNK_COUNT) {
			waveIndex += 1;
			drawWaveform();
		} else {
			waveIndex = 0;
			Button_4_Click();
		}
	}
	void Button_4_Click() {
		std::cout << "Button " << 4 << " clicked." << std::endl;
		if(chunkIndex < 1000) chunkIndex += 1;
		drawWaveform();
	}
	MainWindow(BaseObjectType* cobject, const Glib::RefPtr<Gtk::Builder>&builder):
		Gtk::Window(cobject), 
		_builder(builder),
		waveBuffer(1000, 1, Chunk(CHUNK_COUNT*SAMPLE_COUNT)),
		reader(std::string("../../data/Al_25keV-1.cdb"), &waveBuffer, 1000),
		chunkIndex(0),
		waveIndex(0)
	{
		//Wait for all the waveforms to be loaded.
		fits.load(std::string("results.txt"));
		reader.readToBuffer();
		/* Retrieve all widgets. */
		_builder->get_widget("button1", button1);
		_builder->get_widget("button2", button2);
		_builder->get_widget("button3", button3);
		_builder->get_widget("button4", button4);
		_builder->get_widget("image1", image);
		
		/* Connect signals. */
		button1->signal_clicked().connect(sigc::mem_fun(*this, &MainWindow::Button_1_Click));
		button2->signal_clicked().connect(sigc::mem_fun(*this, &MainWindow::Button_2_Click));
		button3->signal_clicked().connect(sigc::mem_fun(*this, &MainWindow::Button_3_Click));
		button4->signal_clicked().connect(sigc::mem_fun(*this, &MainWindow::Button_4_Click));
		/* Actions. */
		drawWaveform();
//		Glib::RefPtr<Gtk::Action>::cast_dynamic(_builder->get_object("action_quit"))->
//			signal_activate().connect(sigc::mem_fun(*this, &MainWindow::OnQuit));
	}
};

int main(int argc, char **argv)
{
    Gtk::Main app(argc, argv);
    Glib::RefPtr<Gtk::Builder> builder = Gtk::Builder::create_from_file("Viewer.glade");
    MainWindow *mainWindow = 0;
    builder->get_widget_derived("Viewer", mainWindow);
    app.run(*mainWindow);
    delete mainWindow;
    return 0;
}
