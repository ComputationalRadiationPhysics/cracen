# Speichertransfer zwischen Host und Device

## Grobgliederung

Die Eingangsdaten werden aus einem Ringpuffer gelesen. Die Ergebnisse werden zunächst in einem Ringpuffer zwischengespeichert und dann in eine Datei geschrieben. Jeder Node erhält seinen eigenen Thread, der die Daten aus dem Eingangspuffer auf die Grafikkarte kopieren. Das Schreiben in die Datei wird ebenfalls von einem eigenen Thread erledigt. Damit es nicht zu Lese- oder Schreibkonflikten kommt, werden die Puffer durch Semaphoren geschützt. Um die einzelnen Nodes zu verwalten wird eine Klasse implementiert, deren Objekte jeweils einen der 4 Nodes der GPU verwalten.

## Aufgaben

- Initialisierungscode schreiben
	- Auslesen der Systemdetails mit cudaDeviceProps
	- Objekte für jeden Node erstellen
	- die einzelnen Threads starten

- Node Klasse
	- Attribute
		- Device Pointer
		- Referenz auf Ringpuffer für Eingang
		- Referenz auf Ringpuffer für Ausgang
		- finish 
			- Variable die vom Host gesetzt werden kann um anzuzeigen, dass keine neuen Daten mehr folgen
	- Konstruktor/Destruktor
	- Methoden
		- stop (public)
			- Host beendet den Thread evt. auch im Destruktor implementiert
		- run (public)
			- Wird als eigener Thread gestartet
		- copyToDevice (private)
		- getResults (private)

- Speichertransfer Host -> Device
	- In Methode copyToDevice
	- Textur Referenz anlegen (im File scope)
	- Texturobjekt initialisieren
	- Daten in Textur kopieren
	- Anschließend Kernel starten
	
- Speichertransfer Device -> Host
	- Ende des Kernels abwarten
	- cudaMemcopy der Daten auf den Stack
	- Einfügen in Ausgangspuffer
	
- Ausgangspuffer
	- Attribute
		- ofstream outputFile
		- vector<bool> finished 
			- Zeigt für jeden Node an, ob dieser noch weitere Daten liefert
	- Initialisierung
		- Datei öffnen
		- finished mit false füllen
	- Destruktor
		- Datei schließen
	- Empty
		- Wird in eigenen Thread gestartet
		- Läuft, solange mindestens einer der Werte aus dem vector finished false
		- Schreibt die Ergebnisse unsortiert in Datei
		- Sortierte Ausgabe mögliche Erweiterung 
	- Write
		- Id und Datenstruct in Queue schreiben
	- finished
		- setzt ein Bit für einen Node in finished auf true
