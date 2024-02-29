const gulp = require('gulp');
const uglify = require('gulp-uglify');

gulp.task('minify', function () {
    return gulp.src('firebase-database.js') // Replace 'yourFileName.js' with the actual filename
        .pipe(uglify())
        .pipe(gulp.dest('dist')); // Output directory for the minified file
});
